param(
    [int]$ShardCount = 6
)

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$harness = Join-Path $workspace "baselines\mgr-harness"
$total = 0

for ($shard = 0; $shard -lt $ShardCount; $shard++) {
    $name = "mgr-paired2-$shard"
    $file = Join-Path $harness "results\mgr_clear320_final_shard${shard}of${ShardCount}.jsonl"
    $count = if (Test-Path -LiteralPath $file) {
        (Get-Content -LiteralPath $file | Measure-Object -Line).Lines
    } else {
        0
    }
    $total += $count
    $status = docker inspect -f "{{.State.Status}}:{{.State.ExitCode}}" $name 2>$null
    Write-Host "shard $shard`: $count records, $status"
}

Write-Host "total: $total/320"
