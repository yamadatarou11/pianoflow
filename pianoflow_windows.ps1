Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

[System.Windows.Forms.Application]::EnableVisualStyles()

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$hqPython = Join-Path $root '.venv-pianoflow-hq\Scripts\python.exe'
if (Test-Path -LiteralPath $hqPython) {
    & $hqPython -c "import faster_whisper, janome" 2>$null
    if ($LASTEXITCODE -ne 0) {
        [System.Windows.Forms.MessageBox]::Show('初回のみ、日本語文字起こし機能を準備します。数分かかる場合があります。','PianoFlow 初回準備','OK','Information') | Out-Null
        & $hqPython -m pip install 'faster-whisper==1.2.1' 'Janome==0.5.0'
    }
}
$ffmpegCommand = Get-Command 'ffmpeg.exe' -ErrorAction SilentlyContinue
$ffmpeg = if ($ffmpegCommand) { $ffmpegCommand.Source } else { $null }
if (-not $ffmpeg) {
    $wingetRoot = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'
    $ffmpeg = Get-ChildItem -LiteralPath $wingetRoot -Filter 'ffmpeg.exe' -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like '*Gyan.FFmpeg*\bin\ffmpeg.exe' } |
        Select-Object -First 1 -ExpandProperty FullName
}
$script:source = $null
$script:prepared = $null
$script:server = $null
$studioPort = 4194
$expectedAnalysisVersion = 39
$expectedServerVersion = 126

function Stop-PianoFlowPortListeners {
    $pattern = '^\s*TCP\s+\S+:' + $studioPort + '\s+\S+\s+LISTENING\s+(\d+)\s*$'
    $listenerPids = @(
        & netstat.exe -ano -p TCP 2>$null |
            ForEach-Object {
                $match = [regex]::Match($_, $pattern)
                if ($match.Success) { [int]$match.Groups[1].Value }
            } |
            Sort-Object -Unique
    )
    foreach ($listenerPid in $listenerPids) {
        Stop-Process -Id $listenerPid -Force -ErrorAction SilentlyContinue
    }
    if ($listenerPids.Count) { Start-Sleep -Milliseconds 700 }
}

function Color([string]$hex) { [System.Drawing.ColorTranslator]::FromHtml($hex) }

$form = New-Object System.Windows.Forms.Form
$form.Text = 'PianoFlow Windows'
$form.Size = New-Object System.Drawing.Size(760,665)
$form.MinimumSize = New-Object System.Drawing.Size(680,625)
$form.StartPosition = 'CenterScreen'
$form.BackColor = Color '#0a0a12'
$form.ForeColor = Color '#f5f2ff'
$form.Font = New-Object System.Drawing.Font('Yu Gothic UI',10)

$logo = New-Object System.Windows.Forms.Label
$logo.Text = 'P'
$logo.Location = New-Object System.Drawing.Point(38,28)
$logo.Size = New-Object System.Drawing.Size(38,38)
$logo.TextAlign = 'MiddleCenter'
$logo.BackColor = Color '#7566f5'
$logo.ForeColor = [System.Drawing.Color]::White
$logo.Font = New-Object System.Drawing.Font('Georgia',18,[System.Drawing.FontStyle]::Bold)
$form.Controls.Add($logo)

$brand = New-Object System.Windows.Forms.Label
$brand.Text = 'PianoFlow  WINDOWS'
$brand.Location = New-Object System.Drawing.Point(88,34)
$brand.AutoSize = $true
$brand.Font = New-Object System.Drawing.Font('Yu Gothic UI',16,[System.Drawing.FontStyle]::Bold)
$form.Controls.Add($brand)

$privacy = New-Object System.Windows.Forms.Label
$privacy.Text = '● 端末内だけで処理'
$privacy.Location = New-Object System.Drawing.Point(570,40)
$privacy.AutoSize = $true
$privacy.ForeColor = Color '#74dbb4'
$privacy.Font = New-Object System.Drawing.Font('Yu Gothic UI',9)
$form.Controls.Add($privacy)

$title = New-Object System.Windows.Forms.Label
$title.Text = 'MOV変換からピアノ動画作成まで。'
$title.Location = New-Object System.Drawing.Point(38,92)
$title.Size = New-Object System.Drawing.Size(680,40)
$title.TextAlign = 'MiddleCenter'
$title.Font = New-Object System.Drawing.Font('Yu Gothic UI',22,[System.Drawing.FontStyle]::Bold)
$form.Controls.Add($title)

$subtitle = New-Object System.Windows.Forms.Label
$subtitle.Text = 'MOV / MP4 / MP3 / WAV に対応。MOVはYouTube互換MP4へ自動変換します。'
$subtitle.Location = New-Object System.Drawing.Point(38,135)
$subtitle.Size = New-Object System.Drawing.Size(680,25)
$subtitle.TextAlign = 'MiddleCenter'
$subtitle.ForeColor = Color '#9791aa'
$subtitle.Font = New-Object System.Drawing.Font('Yu Gothic UI',9)
$form.Controls.Add($subtitle)

$card = New-Object System.Windows.Forms.Panel
$card.Location = New-Object System.Drawing.Point(42,180)
$card.Size = New-Object System.Drawing.Size(660,155)
$card.BackColor = Color '#141420'
$card.BorderStyle = 'FixedSingle'
$form.Controls.Add($card)

$fileLabel = New-Object System.Windows.Forms.Label
$fileLabel.Text = '動画または音源を選択してください'
$fileLabel.Location = New-Object System.Drawing.Point(20,23)
$fileLabel.Size = New-Object System.Drawing.Size(618,28)
$fileLabel.TextAlign = 'MiddleCenter'
$fileLabel.Font = New-Object System.Drawing.Font('Yu Gothic UI',11,[System.Drawing.FontStyle]::Bold)
$card.Controls.Add($fileLabel)

$fileInfo = New-Object System.Windows.Forms.Label
$fileInfo.Text = 'MOV / MP4 / MP3 / WAV / M4A / OGG'
$fileInfo.Location = New-Object System.Drawing.Point(20,55)
$fileInfo.Size = New-Object System.Drawing.Size(618,22)
$fileInfo.TextAlign = 'MiddleCenter'
$fileInfo.ForeColor = Color '#777287'
$fileInfo.Font = New-Object System.Drawing.Font('Yu Gothic UI',9)
$card.Controls.Add($fileInfo)

$choose = New-Object System.Windows.Forms.Button
$choose.Text = 'ファイルを選択'
$choose.Location = New-Object System.Drawing.Point(245,94)
$choose.Size = New-Object System.Drawing.Size(170,39)
$choose.FlatStyle = 'Flat'
$choose.BackColor = Color '#302b59'
$choose.ForeColor = Color '#dcd7ff'
$card.Controls.Add($choose)

$progress = New-Object System.Windows.Forms.ProgressBar
$progress.Location = New-Object System.Drawing.Point(52,355)
$progress.Size = New-Object System.Drawing.Size(640,15)
$progress.Style = 'Continuous'
$form.Controls.Add($progress)

$status = New-Object System.Windows.Forms.Label
$status.Text = '音源なしでもPianoFlowを開けます'
$status.Location = New-Object System.Drawing.Point(42,380)
$status.Size = New-Object System.Drawing.Size(660,25)
$status.TextAlign = 'MiddleCenter'
$status.ForeColor = Color '#817b90'
$status.Font = New-Object System.Drawing.Font('Yu Gothic UI',9)
$form.Controls.Add($status)

$prepare = New-Object System.Windows.Forms.Button
$prepare.Text = '変換・準備する'
$prepare.Location = New-Object System.Drawing.Point(190,420)
$prepare.Size = New-Object System.Drawing.Size(175,48)
$prepare.FlatStyle = 'Flat'
$prepare.BackColor = Color '#7465f4'
$prepare.ForeColor = [System.Drawing.Color]::White
$prepare.Enabled = $false
$prepare.Font = New-Object System.Drawing.Font('Yu Gothic UI',10,[System.Drawing.FontStyle]::Bold)
$form.Controls.Add($prepare)

$open = New-Object System.Windows.Forms.Button
$open.Text = 'PianoFlowを開く'
$open.Location = New-Object System.Drawing.Point(380,420)
$open.Size = New-Object System.Drawing.Size(175,48)
$open.FlatStyle = 'Flat'
$open.BackColor = Color '#d65bc2'
$open.ForeColor = [System.Drawing.Color]::White
$open.Enabled = $true
$open.Font = New-Object System.Drawing.Font('Yu Gothic UI',10,[System.Drawing.FontStyle]::Bold)
$form.Controls.Add($open)

$steps = New-Object System.Windows.Forms.Label
$steps.Text = '1  ファイル選択  →  2  MOVなら自動変換  →  3  PianoFlowで解析  →  4  完成動画を保存'
$steps.Location = New-Object System.Drawing.Point(42,492)
$steps.Size = New-Object System.Drawing.Size(660,42)
$steps.TextAlign = 'MiddleCenter'
$steps.BackColor = Color '#11111b'
$steps.ForeColor = Color '#aaa4bc'
$steps.BorderStyle = 'FixedSingle'
$steps.Font = New-Object System.Drawing.Font('Yu Gothic UI',9,[System.Drawing.FontStyle]::Bold)
$form.Controls.Add($steps)

$iphone = New-Object System.Windows.Forms.Button
$iphone.Text = '完成WebMをiPhone対応MP4へ変換'
$iphone.Location = New-Object System.Drawing.Point(190,548)
$iphone.Size = New-Object System.Drawing.Size(365,45)
$iphone.FlatStyle = 'Flat'
$iphone.BackColor = Color '#1f685d'
$iphone.ForeColor = [System.Drawing.Color]::White
$iphone.Font = New-Object System.Drawing.Font('Yu Gothic UI',10,[System.Drawing.FontStyle]::Bold)
$form.Controls.Add($iphone)

$iphoneHelp = New-Object System.Windows.Forms.Label
$iphoneHelp.Text = 'PianoFlowで保存した .webm を選択すると、iPhone・写真アプリ・YouTube対応MP4に変換します。'
$iphoneHelp.Location = New-Object System.Drawing.Point(42,600)
$iphoneHelp.Size = New-Object System.Drawing.Size(660,22)
$iphoneHelp.TextAlign = 'MiddleCenter'
$iphoneHelp.ForeColor = Color '#696478'
$iphoneHelp.Font = New-Object System.Drawing.Font('Yu Gothic UI',8)
$form.Controls.Add($iphoneHelp)

$choose.Add_Click({
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = '動画または音源を選択'
    $dialog.Filter = '対応ファイル|*.mov;*.mp4;*.mp3;*.wav;*.m4a;*.ogg|すべてのファイル|*.*'
    if ($dialog.ShowDialog() -ne 'OK') { return }
    $script:source = Get-Item -LiteralPath $dialog.FileName
    $script:prepared = $null
    $fileLabel.Text = $script:source.Name
    $fileInfo.Text = ('{0}  •  {1:N1} MB' -f $script:source.Extension.ToUpper(), ($script:source.Length / 1MB))
    $progress.Value = 0
    $open.Enabled = $true
    $prepare.Enabled = $true
    if ($script:source.Extension -ieq '.mov') {
        $prepare.Text = 'MOVをMP4へ変換'
        $status.Text = 'MOVをMP4へ変換する準備ができました'
        $status.ForeColor = Color '#b7afff'
    } else {
        $prepare.Text = 'PianoFlowで使う準備'
        $status.Text = 'このファイルはそのまま使用できます'
        $status.ForeColor = Color '#79ddb8'
    }
})

$prepare.Add_Click({
    if (-not $script:source) { return }
    if ($script:source.Extension -ine '.mov') {
        $script:prepared = $script:source.FullName
        $progress.Value = 100
        $open.Enabled = $true
        $status.Text = '準備完了。「PianoFlowを開く」を押してください'
        $status.ForeColor = Color '#79ddb8'
        return
    }
    if (-not $ffmpeg -or -not (Test-Path -LiteralPath $ffmpeg)) {
        [System.Windows.Forms.MessageBox]::Show('FFmpegが見つかりません。PCを再起動してもう一度お試しください。','エラー','OK','Error') | Out-Null
        return
    }
    $script:prepared = Join-Path $script:source.DirectoryName ($script:source.BaseName + '_pianoflow.mp4')
    if ((Test-Path -LiteralPath $script:prepared) -and [System.Windows.Forms.MessageBox]::Show('変換済みMP4を上書きしますか？','確認','YesNo','Question') -ne 'Yes') { return }
    $prepare.Enabled = $false
    $open.Enabled = $false
    $status.Text = 'MOVを変換しています。しばらくお待ちください…'
    $status.ForeColor = Color '#b7afff'
    $progress.Style = 'Marquee'
    $form.Refresh()
    $arguments = @('-y','-i',$script:source.FullName,'-map_metadata','0','-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2','-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',$script:prepared)
    $logDir = Join-Path $root 'work'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $errorLog = Join-Path $logDir 'mov-convert-last-error.log'
    # Start-Process -ArgumentList rejoins arguments into one string on Windows,
    # which breaks input/output paths containing spaces or parentheses. Invoke
    # FFmpeg directly with an argument array so every path stays one argument.
    & $ffmpeg @arguments 2> $errorLog
    $convertExitCode = $LASTEXITCODE
    $progress.Style = 'Continuous'
    if ($convertExitCode -eq 0 -and (Test-Path -LiteralPath $script:prepared)) {
        $progress.Value = 100
        $prepare.Enabled = $true
        $prepare.Text = 'もう一度変換'
        $open.Enabled = $true
        $status.Text = 'MP4変換完了。PianoFlowを開けます'
        $status.ForeColor = Color '#79ddb8'
        [System.Windows.Forms.MessageBox]::Show("変換が完了しました。`n`n$script:prepared",'準備完了','OK','Information') | Out-Null
    } else {
        $prepare.Enabled = $true
        $status.Text = 'MOVの変換に失敗しました'
        $status.ForeColor = Color '#ff849a'
        $detail = ''
        if (Test-Path -LiteralPath $errorLog) {
            $detail = (Get-Content -LiteralPath $errorLog -Tail 8 -ErrorAction SilentlyContinue) -join "`n"
        }
        $message = "このMOVを変換できませんでした。"
        if ($detail) { $message += "`n`n詳細:`n$detail" }
        [System.Windows.Forms.MessageBox]::Show($message,'変換エラー','OK','Error') | Out-Null
    }
})

$open.Add_Click({
    if ($script:prepared) {
        Set-Clipboard -Value $script:prepared
    }
    $serverReady = $false
    try {
        $existingCapability = Invoke-RestMethod -Uri ("http://localhost:{0}/capabilities" -f $studioPort) -TimeoutSec 3
        $serverReady = ($existingCapability.aiMelody -and [int]$existingCapability.analysisVersion -eq $expectedAnalysisVersion -and [int]$existingCapability.version -eq $expectedServerVersion)
        if (-not $serverReady) {
            Stop-PianoFlowPortListeners
            $script:server = $null
        }
    } catch {
        # No usable listener. A stale duplicate can still own the port without
        # answering HTTP, so clear listeners before starting the current server.
        Stop-PianoFlowPortListeners
        $script:server = $null
    }
    if (-not $serverReady -and (-not $script:server -or $script:server.HasExited)) {
        $serverScript = Join-Path $root 'pianoflow_server.py'
        $aiPython = Join-Path $root '.venv-pianoflow\Scripts\python.exe'
        if (-not (Test-Path -LiteralPath $aiPython)) {
            [System.Windows.Forms.MessageBox]::Show('PianoFlow AI環境が見つかりません。','起動エラー','OK','Error') | Out-Null
            return
        }
        $script:server = Start-Process -FilePath $aiPython -ArgumentList @($serverScript,'--port',$studioPort,'--ffmpeg',$ffmpeg) -PassThru -WindowStyle Hidden
        Start-Sleep -Milliseconds 900
    }
    try {
        $capability = Invoke-RestMethod -Uri ("http://localhost:{0}/capabilities" -f $studioPort) -TimeoutSec 5
        if (-not $capability.aiMelody -or [int]$capability.analysisVersion -ne $expectedAnalysisVersion -or [int]$capability.version -ne $expectedServerVersion) {
            throw '現在のメロディー解析機能を確認できません'
        }
    } catch {
        [System.Windows.Forms.MessageBox]::Show('メロディー解析サーバーを起動できませんでした。Windows版を閉じて、もう一度起動してください。','起動エラー','OK','Error') | Out-Null
        return
    }
    # Open a one-time file-origin bridge first. It can read the settings saved
    # by direct HTML use, copies them to the local server, then immediately
    # redirects to the normal Windows/server page.
    $htmlPath = Join-Path $root 'PianoFlow_v126.html'
    $fileUrl = 'file:///' + (($htmlPath -replace '\\','/') -replace ' ','%20')
    Start-Process ("{0}?pianoflow-transfer=1&port={1}&t={2}" -f $fileUrl, $studioPort, [DateTimeOffset]::Now.ToUnixTimeSeconds())
    if ($script:prepared) {
        [System.Windows.Forms.MessageBox]::Show("PianoFlowを開きました。`n`n解析する場合は「音源を選ぶ」から次のファイルを選択してください。`n`n$script:prepared`n`nファイルの場所はコピー済みです。",'PianoFlowを開きました','OK','Information') | Out-Null
    } else {
        [System.Windows.Forms.MessageBox]::Show('PianoFlowを開きました。音源の解析、学習結果の確認、ピッチ編集を必要なときに選べます。','PianoFlowを開きました','OK','Information') | Out-Null
    }
})

$iphone.Add_Click({
    if (-not $ffmpeg -or -not (Test-Path -LiteralPath $ffmpeg)) {
        [System.Windows.Forms.MessageBox]::Show('動画変換エンジンが見つかりません。','エラー','OK','Error') | Out-Null
        return
    }
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = 'PianoFlowで完成したWebM動画を選択'
    $dialog.Filter = 'WebM動画|*.webm|すべての動画|*.webm;*.mov;*.mp4'
    if ($dialog.ShowDialog() -ne 'OK') { return }
    $inputVideo = Get-Item -LiteralPath $dialog.FileName
    $outputVideo = Join-Path $inputVideo.DirectoryName ($inputVideo.BaseName + '_iPhone.mp4')
    if ((Test-Path -LiteralPath $outputVideo) -and [System.Windows.Forms.MessageBox]::Show('iPhone用MP4を上書きしますか？','確認','YesNo','Question') -ne 'Yes') { return }

    $iphone.Enabled = $false
    $progress.Style = 'Marquee'
    $status.Text = 'iPhone対応MP4へ変換しています…'
    $status.ForeColor = Color '#84ddd0'
    $form.Refresh()
    $convertArgs = @(
        '-y','-i',$inputVideo.FullName,
        '-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p',
        '-c:a','aac','-b:a','192k','-ar','48000','-movflags','+faststart',
        $outputVideo
    )
    & $ffmpeg @convertArgs 2>$null
    $exitCode = $LASTEXITCODE
    $progress.Style = 'Continuous'
    $iphone.Enabled = $true
    if ($exitCode -eq 0 -and (Test-Path -LiteralPath $outputVideo)) {
        $progress.Value = 100
        $status.Text = 'iPhone対応MP4が完成しました'
        $status.ForeColor = Color '#79ddb8'
        $answer = [System.Windows.Forms.MessageBox]::Show("iPhone対応MP4を保存しました。`n`n$outputVideo`n`n保存フォルダを開きますか？",'変換完了','YesNo','Information')
        if ($answer -eq 'Yes') { Start-Process explorer.exe -ArgumentList @('/select,', $outputVideo) }
    } else {
        $status.Text = 'iPhone対応MP4への変換に失敗しました'
        $status.ForeColor = Color '#ff849a'
        [System.Windows.Forms.MessageBox]::Show('WebMをMP4へ変換できませんでした。','変換エラー','OK','Error') | Out-Null
    }
})

$form.Add_FormClosed({ if ($script:server -and -not $script:server.HasExited) { Stop-Process -Id $script:server.Id -Force -ErrorAction SilentlyContinue } })
[void]$form.ShowDialog()
