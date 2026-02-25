
[Console]::InputEncoding  = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null


$SRC_ROOT = "C:\LG_Chemistry\data\raw"
$DST_ROOT = "C:\LG_Chemistry\data\processed"
$NAS_ROOT = "\\infrax_nas_sub\LG_Chemistry\inspection\defect\segment\datasets\processed"

$LINE  = "SSBR"
$GRADE = "unknown"
$BATCH_SIZE = 9
$RESUME_THRESHOLD = 80   # 진행률 기준값 (여기만 바꾸면 됨)

$VENV_PYTHON = "D:\04_BadRubber-Inspector-DefectDetection\venv_py3.10\Scripts\python.exe"
$PY_SCRIPT   = "D:\04_BadRubber-Inspector-DefectDetection\report_auto.py"

Clear-Host
Write-Host "======================================"
Write-Host " [$LINE] Processing started"
Write-Host " Start Time: $(Get-Date)"
Write-Host "======================================"

# 나스 업로드
function Upload-ToNAS($localPath, $nasPath) {

    if (!(Test-Path $localPath)) {
        return $false
    }

    if (!(Test-Path $nasPath)) {
        New-Item -ItemType Directory -Path $nasPath -Force | Out-Null
    }

    robocopy $localPath $nasPath /E /XO /Z /R:3 /W:5 | Out-Null

    return ($LASTEXITCODE -le 7)
}

# 나스 업로드 후 처리
function Cleanup-AfterUpload($date, $doneFlag, $rawPath, $dstPath, $processingJson, $deleteJson = $true) {

    if ($deleteJson -and (Test-Path $processingJson)) {
        Remove-Item $processingJson -Force
    }

    New-Item -Path $doneFlag -ItemType File -Force | Out-Null

    if (Test-Path $rawPath) {
        Remove-Item $rawPath -Recurse -Force
        Write-Host " RAW Delete Complete"
    }

    if (Test-Path $dstPath) {
        Remove-Item $dstPath -Recurse -Force
        Write-Host " DST Folder Delete Complete"
    }

    Write-Host " $date Complete"
}

# 파이썬 실행
function Run-PythonProcess($date) {

    & $VENV_PYTHON $PY_SCRIPT `
        --src-root $SRC_ROOT `
        --dst-root $DST_ROOT `
        --line $LINE `
        --grade $GRADE `
        --dates $date `
        --batch-size $BATCH_SIZE

    return ($LASTEXITCODE -eq 0)
}

# 메인 루프
while ($true) {

    try {
        $linePath = Join-Path $SRC_ROOT $LINE
        $dateFolders = Get-ChildItem $linePath -Directory |
                       Where-Object { $_.Name -match '^\d{4}-\d{2}-\d{2}$' }

        if ($dateFolders.Count -eq 0) {
            Start-Sleep -Seconds 60
            continue
        }
        foreach ($folder in $dateFolders) {

            $date = $folder.Name
            $doneFlag = Join-Path (Join-Path (Join-Path $DST_ROOT $LINE) $GRADE) "$date.done"
            $processingJson = Join-Path (Join-Path (Join-Path $DST_ROOT $LINE) $GRADE) "$date.processing.json"
            $localProcessedPath = Join-Path (Join-Path (Join-Path $DST_ROOT $LINE) $GRADE) $date
            $nasTargetPath = Join-Path (Join-Path (Join-Path $DST_ROOT $LINE) $GRADE) $date
            $rawFolderPath = Join-Path $linePath $date

            if (Test-Path $doneFlag) {
                continue
            }

            if (Test-Path $processingJson) {

                $progress = Get-Content $processingJson | ConvertFrom-Json

                Write-Host "Processing: $($progress.progress_percent)%"

                if ($progress.progress_percent -gt $RESUME_THRESHOLD) {

                    Write-Host "Skip reprocessing and NAS upload"

                    if (Upload-ToNAS $localProcessedPath $nasTargetPath) {
                        Cleanup-AfterUpload `
                            -date $date `
                            -doneFlag $doneFlag `
                            -rawPath $rawFolderPath `
                            -dstPath $localProcessedPath `
                            -processingJson $processingJson `
                            -deleteJson $false
                    }
                    else {
                        Write-Host " NAS Upload Failed"
                    }

                    continue
                }
                else {
                    Write-Host "reprocessing start"
                    #Remove-Item $processingJson -Force
                }
            }

            Write-Host ""
            Write-Host "---------------------------------------"
            Write-Host " $date Processing started"
            Write-Host " Start Time: $(Get-Date)"

            if (Run-PythonProcess $date) {

                if (Upload-ToNAS $localProcessedPath $nasTargetPath) {

                    Cleanup-AfterUpload `
                        -date $date `
                        -doneFlag $doneFlag `
                        -rawPath $rawFolderPath `
                        -dstPath $localProcessedPath `
                        -processingJson $processingJson `
                        -deleteJson $true
                }
                else {
                    Write-Host " NAS Upload Failed"
                }
            }

            Write-Host "---------------------------------------"
        }
    }
    catch {
        Write-Host "Unexpected Error: $_"
    }

    Start-Sleep -Seconds 60
}