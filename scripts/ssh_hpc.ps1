# SSH HPC 自动重试包装：隧道断了自动重试（最多 8 次，间隔 10 秒）
# 用法: .\scripts\ssh_hpc.ps1 "你的命令" 或  & .\scripts\ssh_hpc.ps1 -Cmd "squeue -u jiahuiwang24"
param(
    [Parameter(Mandatory = $true)][string]$Cmd,
    [int]$Retries = 8,
    [int]$Delay = 10
)

for ($i = 1; $i -le $Retries; $i++) {
    $output = ssh hpc $Cmd 2>&1
    if ($LASTEXITCODE -eq 0) {
        $output
        exit 0
    }
    Write-Host "[retry $i/$Retries] tunnel unstable, retrying in ${Delay}s..." -ForegroundColor Yellow
    Start-Sleep -Seconds $Delay
}
Write-Error "SSH failed after $Retries retries"
exit 1
