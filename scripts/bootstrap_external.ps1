param(
    [ValidateSet("core","optional","all")]
    [string]$Group = "core"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Dest = Join-Path $Root "external\repos"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

function Clone-IfMissing([string]$Name,[string]$Url) {
    $Path = Join-Path $Dest $Name
    if (Test-Path (Join-Path $Path ".git")) {
        Write-Host "[exists] $Name"
    } else {
        Write-Host "[clone] $Name"
        git clone --depth 1 $Url $Path
    }
}
function Clone-Core {
    Clone-IfMissing "nnsight" "https://github.com/ndif-team/nnsight.git"
    Clone-IfMissing "pyvene" "https://github.com/frankaging/pyvene.git"
    Clone-IfMissing "axbench" "https://github.com/stanfordnlp/axbench.git"
    Clone-IfMissing "internal_probing" "https://github.com/zazamrykh/internal_probing.git"
    Clone-IfMissing "Reasoning-Flow" "https://github.com/MasterZhou1/Reasoning-Flow.git"
    Clone-IfMissing "multi-component-causal-tracing" "https://github.com/ZiruiYan/multi-component-causal-tracing.git"
    Clone-IfMissing "activation-steering" "https://github.com/IBM/activation-steering.git"
}
function Clone-Optional {
    Clone-IfMissing "SAELens" "https://github.com/decoderesearch/SAELens.git"
    Clone-IfMissing "sae-feature-consistency" "https://github.com/xiangchensong/sae-feature-consistency.git"
    Clone-IfMissing "ICR_Probe" "https://github.com/XavierZhang2002/ICR_Probe.git"
    Clone-IfMissing "interpretability" "https://github.com/PAIR-code/interpretability.git"
    Clone-IfMissing "honest_llama" "https://github.com/likenneth/honest_llama.git"
    Clone-IfMissing "pyreft" "https://github.com/stanfordnlp/pyreft.git"
    Clone-IfMissing "StateBridge" "https://github.com/YanwenPneg/StateBridge.git"
    Clone-IfMissing "selective-steering" "https://github.com/knoveleng/steering.git"
}
if ($Group -eq "core") { Clone-Core }
elseif ($Group -eq "optional") { Clone-Optional }
else { Clone-Core; Clone-Optional }

Write-Host "Cloned only. Do NOT install all repos into one environment."
Write-Host "Read docs\EXTERNAL_MODULES.md."
