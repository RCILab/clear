$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$mgrRepo = Join-Path $workspace "baselines\merry-go-round"
$harness = Join-Path $workspace "baselines\mgr-harness"
$instances = Join-Path $workspace "results\mgr_paired_instances"

foreach ($robots in @(20, 40, 60, 80)) {
    $name = "mgr-timing-$robots"
    $existing = docker ps -a `
        --filter "name=^/${name}$" `
        --format "{{.Names}}"
    if ($existing) {
        docker rm -f $name | Out-Null
    }
    $output = "/harness/results/mgr_timing_cvxopt_n${robots}.jsonl"
    $hostOutput = Join-Path $harness `
        "results\mgr_timing_cvxopt_n${robots}.jsonl"
    if (Test-Path -LiteralPath $hostOutput) {
        Remove-Item -LiteralPath $hostOutput
    }
    Get-ChildItem -LiteralPath (Join-Path $harness "results") `
        -Filter "mgr_timing_cvxopt_n${robots}_part*.jsonl" |
        Remove-Item
    docker run -d `
        --name $name `
        -e MGR_QP_BACKEND=cvxopt `
        -e OPENBLAS_NUM_THREADS=1 `
        -e OMP_NUM_THREADS=1 `
        -e MKL_NUM_THREADS=1 `
        -e PYTHONPYCACHEPREFIX=/tmp/pycache `
        --mount "type=bind,source=$mgrRepo,target=/mgr,readonly" `
        --mount "type=bind,source=$harness,target=/harness" `
        --mount "type=bind,source=$instances,target=/paired,readonly" `
        mgr-optimized:latest `
        python /harness/run_batch.py `
        --manifest /paired/manifest.json `
        --output $output `
        --robots $robots `
        --seeds 0 7 13 `
        --audit-pruning `
        --timing-warmup-steps 40 | Out-Null
}
Write-Host "Started four exact-CVXOPT MGR timing audits (12 cases per N)."
