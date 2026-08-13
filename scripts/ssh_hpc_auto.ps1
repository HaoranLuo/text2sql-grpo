# ssh_hpc_auto.ps1 — 自动恢复版 SSH 跳板
# 用法: powershell -File scripts/ssh_hpc_auto.ps1 "squeue"
# 逻辑: ①正常执行命令 ②失败 → 尝试从"端口公告板"拉取最新端口并更新 ~/.ssh/config → 重试
# 公告板: 设置环境变量 TUNNEL_PORT_URL(如 Gist raw 地址, 内容一行 "6.tcp.cpolar.cn:10889")
#         未设置时仅重试 8 次(行为同 ssh_hpc.ps1)并提示手动同步。
param([Parameter(Mandatory=$true)][string]$Cmd)

$sshConfig = "$env:USERPROFILE\.ssh\config"
$maxRetry = 8

function Invoke-SshHpc {
    $output = ssh hpc $Cmd 2>&1
    return ($LASTEXITCODE -eq 0), ($output | Out-String)
}

# 第一次尝试
$ok, $out = Invoke-SshHpc
if ($ok) { $out; exit 0 }

# 失败: 尝试从公告板更新端口
$url = $env:TUNNEL_PORT_URL
if ($url) {
    Write-Host "[auto] ssh 失败, 尝试从公告板同步最新端口..."
    try {
        $raw = (Invoke-WebRequest -Uri $url -TimeoutSec 15 -UseBasicParsing).Content.Trim()
        if ($raw -match '^[A-Za-z0-9.\-]+:(\d+)$') {
            $newPort = $Matches[1]
            $cfg = Get-Content $sshConfig -Raw
            if ($cfg -notmatch "Port $newPort\b") {
                $cfg = $cfg -replace 'Port \d+', "Port $newPort"
                Set-Content $sshConfig $cfg -Encoding ascii
                Write-Host "[auto] 端口已更新为 $newPort"
            }
        } else {
            Write-Host "[auto] 公告板内容无法解析: $raw"
        }
    } catch {
        Write-Host "[auto] 公告板拉取失败: $($_.Exception.Message)"
    }
}

# 重试(端口更新后或未更新都重试)
for ($i = 1; $i -le $maxRetry; $i++) {
    Start-Sleep -Seconds 10
    $ok, $out = Invoke-SshHpc
    if ($ok) { $out; Write-Host "[auto] 第 $i 次重试成功"; exit 0 }
    Write-Host "[auto] 第 $i 次重试失败"
}
Write-Error "SSH 恢复失败($maxRetry 次重试)。建议: ①确认 gf-bastion 在线 ②问堡垒机 AI 拿新端口并同步 ~/.ssh/config ③手动设置 TUNNEL_PORT_URL 后重跑"
exit 1
