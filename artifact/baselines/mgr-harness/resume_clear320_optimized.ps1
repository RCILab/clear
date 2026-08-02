param(
    [int]$ShardCount = 12
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$mgrRepo = Join-Path $workspace "baselines\merry-go-round"
$harness = Join-Path $workspace "baselines\mgr-harness"
$instances = Join-Path $workspace "results\mgr_paired_instances"

for ($shard = 0; $shard -lt $ShardCount; $shard++) {
    $name = "mgr-opt-$shard"
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

    $output = "/harness/results/mgr_clear320_final_cont_shard${shard}of${ShardCount}.jsonl"
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
        --completed-glob "/harness/results/mgr_clear320_final_shard*of6.jsonl" `
        --shard-index $shard `
        --shard-count $ShardCount `
        --audit-pruning | Out-Null
}

Write-Host "Started or resumed $ShardCount optimized continuation shards."
& (Join-Path $PSScriptRoot "status_clear320_optimized.ps1") `
    -ShardCount $ShardCount
