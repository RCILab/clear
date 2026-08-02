param(
    [int]$ShardCount = 6
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$mgrRepo = Join-Path $workspace "baselines\merry-go-round"
$harness = Join-Path $workspace "baselines\mgr-harness"
$instances = Join-Path $workspace "results\mgr_paired_instances"

for ($shard = 0; $shard -lt $ShardCount; $shard++) {
    $name = "mgr-paired2-$shard"
    $existing = docker ps -a `
        --filter "name=^/${name}$" `
        --format "{{.Names}}"
    if ($existing) {
        $status = docker inspect -f "{{.State.Status}}" $name
        if ($status -ne "running") {
            docker start $name | Out-Null
        }
        continue
    }

    $output = "/harness/results/mgr_clear320_final_shard${shard}of${ShardCount}.jsonl"
    docker run -d `
        --name $name `
        -e OPENBLAS_NUM_THREADS=1 `
        -e OMP_NUM_THREADS=1 `
        -e MKL_NUM_THREADS=1 `
        --mount "type=bind,source=$mgrRepo,target=/mgr,readonly" `
        --mount "type=bind,source=$harness,target=/harness" `
        --mount "type=bind,source=$instances,target=/paired,readonly" `
        mgr-official:84aaefd `
        python /harness/run_batch.py `
        --manifest /paired/manifest.json `
        --output $output `
        --shard-index $shard `
        --shard-count $ShardCount | Out-Null
}

Write-Host "Resumed $ShardCount shards."
Write-Host "Progress:"
for ($shard = 0; $shard -lt $ShardCount; $shard++) {
    $name = "mgr-paired2-$shard"
    $file = Join-Path $harness "results\mgr_clear320_final_shard${shard}of${ShardCount}.jsonl"
    $count = if (Test-Path -LiteralPath $file) {
        (Get-Content -LiteralPath $file | Measure-Object -Line).Lines
    } else {
        0
    }
    $status = docker inspect -f "{{.State.Status}}" $name
    Write-Host "  shard $shard`: $count records, $status"
}
