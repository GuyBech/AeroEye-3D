Add-Type -AssemblyName System.Speech
$dir = $PSScriptRoot
New-Item -ItemType Directory -Force -Path "$dir\vo" | Out-Null
$lines = @{
  'scan'   = 'Scanning.'
  'detect' = 'Target detected.'
  'lock'   = 'Target locked.'
  'solve'  = 'Computing position. Altitude. Heading.'
  'cmd'    = 'Command sent.'
  'laser'  = 'Laser on.'
  'hit'    = 'Target hit.'
}
foreach ($k in $lines.Keys) {
  $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
  $s.SelectVoice('Microsoft David Desktop')
  $s.Rate = -1
  $s.SetOutputToWaveFile("$dir\vo\$k.wav")
  $s.Speak($lines[$k])
  $s.Dispose()
}
Get-ChildItem "$dir\vo" | ForEach-Object { $_.Name + ' ' + $_.Length }
