# cpolar watchdog for the girlfriend's PC (bastion machine)
# Checks whether cpolar is running; restarts it if dead. Logs to cpolar_watchdog.log.
# Install as a scheduled task:
#   schtasks /Create /TN "cpolar_watchdog" /SC MINUTE /MO 5 /TR "powershell -ExecutionPolicy Bypass -WindowStyle Hidden -File C:\cpolar_watchdog.ps1" /F
$log = Join-Path $env:USERPROFILE "cpolar_watchdog.log"
$proc = Get-Process -Name cpolar -ErrorAction SilentlyContinue
if ($proc) {
    exit 0
}
$candidates = @(
    "$env:ProgramFiles\cpolar\cpolar.exe",
    "${env:ProgramFiles(x86)}\cpolar\cpolar.exe",
    "$env:LOCALAPPDATA\cpolar\cpolar.exe",
    "$env:USERPROFILE\AppData\Local\Programs\cpolar\cpolar.exe",
    "C:\cpolar\cpolar.exe"
)
$exe = $null
foreach ($c in $candidates) {
    if (Test-Path $c) { $exe = $c; break }
}
if (-not $exe) {
    $found = Get-Command cpolar -ErrorAction SilentlyContinue
    if ($found) { $exe = $found.Source }
}
if ($exe) {
    Start-Process -FilePath $exe -WindowStyle Hidden
    Add-Content -Path $log -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') cpolar restarted from $exe"
} else {
    Add-Content -Path $log -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') cpolar not found, manual check needed"
}
