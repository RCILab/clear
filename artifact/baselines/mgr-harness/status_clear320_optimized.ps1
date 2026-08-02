param(
    [int]$ShardCount = 12
)

$harness = $PSScriptRoot
$original = 0
Get-ChildItem -LiteralPath (Join-Path $harness "results") `
    -Filter "mgr_clear320_final_shard*of6.jsonl" | ForEach-Object {
        $original += (Get-Content -LiteralPath $_.FullName |
            Measure-Object -Line).Lines
    }

$continuation = 0
for ($shard = 0; $shard -lt $ShardCount; $shard++) {
    $name = "mgr-opt-$shard"
    $file = Join-Path $harness `
        "results\mgr_clear320_final_cont_shard${shard}of${ShardCount}.jsonl"
    $count = if (Test-Path -LiteralPath $file) {
        (Get-Content -LiteralPath $file | Measure-Object -Line).Lines
    } else {
        0
    }
    $continuation += $count
    $status = docker inspect -f "{{.State.Status}}:{{.State.ExitCode}}" `
        $name 2>$null
    Write-Host "continuation shard $shard`: $count records, $status"
}

Write-Host "original corrected records: $original"
Write-Host "optimized continuation: $continuation"
Write-Host "unique target total: $($original + $continuation)/320"
