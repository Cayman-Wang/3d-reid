param(
    [string]$RuntimeRoot = 'D:\node01_spin_runtime_ascii',
    [string]$ReconRoot = 'D:\grad_project_recon_ascii',
    [string]$AssetSourceRoot = 'D:\grad_project_ascii'
)

$ErrorActionPreference = 'Stop'

function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Ensure-Junction {
    param(
        [string]$LinkPath,
        [string]$TargetPath
    )

    if (-not (Test-Path $TargetPath)) {
        throw "Target path does not exist: $TargetPath"
    }

    if (Test-Path $LinkPath) {
        $item = Get-Item $LinkPath -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            $actualTarget = ''
            try {
                $actualTarget = [string]$item.Target
            } catch {
                $actualTarget = ''
            }
            if ($actualTarget -eq $TargetPath) {
                return
            }
            throw "Existing junction has different target: $LinkPath -> $actualTarget (expected $TargetPath)"
        }
        throw "Path already exists and is not a junction: $LinkPath"
    }

    New-Item -ItemType Junction -Path $LinkPath -Target $TargetPath | Out-Null
}

$runtimeMvp = Join-Path $RuntimeRoot 'mvp-demo'
$runtimeAssets = Join-Path $runtimeMvp 'assets'

Ensure-Directory $RuntimeRoot
Ensure-Directory $runtimeMvp
Ensure-Directory $runtimeAssets
Ensure-Directory (Join-Path $runtimeMvp 'data')
Ensure-Directory (Join-Path $runtimeMvp 'output')

Ensure-Junction (Join-Path $RuntimeRoot 'research') (Join-Path $ReconRoot 'research')
Ensure-Junction (Join-Path $runtimeMvp 'scripts') (Join-Path $ReconRoot 'mvp-demo\scripts')
Ensure-Junction (Join-Path $runtimeAssets 'scene') (Join-Path $ReconRoot 'mvp-demo\assets\scene')
Ensure-Junction (Join-Path $runtimeAssets 'models') (Join-Path $AssetSourceRoot 'mvp-demo\assets\models')

Write-Host "Runtime ready: $RuntimeRoot"
Write-Host "  research -> $(Join-Path $ReconRoot 'research')"
Write-Host "  scripts  -> $(Join-Path $ReconRoot 'mvp-demo\scripts')"
Write-Host "  scene    -> $(Join-Path $ReconRoot 'mvp-demo\assets\scene')"
Write-Host "  models   -> $(Join-Path $AssetSourceRoot 'mvp-demo\assets\models')"
