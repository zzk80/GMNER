[CmdletBinding()]
param(
    [string]$Server = "server4090",
    [string]$RemoteProject = "/home/zzk/gmner",
    [string]$LocalRoot = (
        Join-Path $PSScriptRoot "../knowledge/null_release_oof/roberta128"
    ),
    [int[]]$Folds = (0..9),
    [switch]$Watch,
    [ValidateRange(30, 86400)]
    [int]$PollSeconds = 300
)

$ErrorActionPreference = "Stop"
$LocalRoot = [IO.Path]::GetFullPath($LocalRoot)
$RemoteProject = $RemoteProject.TrimEnd("/")
New-Item -ItemType Directory -Path $LocalRoot -Force | Out-Null

function Write-ProgressLog {
    param([string]$Message)
    [Console]::Out.WriteLine(
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    )
}

function Assert-UnderLocalRoot {
    param([string]$Path)

    $resolved = [IO.Path]::GetFullPath($Path)
    $prefix = $LocalRoot.TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    ) + [IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith(
        $prefix,
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing filesystem operation outside local OOF root: $resolved"
    }
}

function Get-ExpectedFeatureHash {
    param([string]$FoldDirectory)

    $checksumPath = Join-Path $FoldDirectory "heldout_features.pt.sha256"
    if (-not (Test-Path -LiteralPath $checksumPath -PathType Leaf)) {
        throw "Missing checksum file: $checksumPath"
    }
    $line = (Get-Content -LiteralPath $checksumPath -TotalCount 1).Trim()
    $expected = ($line -split "\s+")[0].ToLowerInvariant()
    if ($expected -notmatch "^[0-9a-f]{64}$") {
        throw "Invalid SHA-256 in ${checksumPath}: $expected"
    }
    return $expected
}

function Test-LocalFold {
    param(
        [int]$Fold,
        [string]$Directory
    )

    $featurePath = Join-Path $Directory "heldout_features.pt"
    $proofPath = Join-Path $Directory "fold_proof.json"
    $pipelinePath = Join-Path $Directory "pipeline_manifest.json"
    $archivePath = Join-Path $Directory "fold_archive_manifest.json"
    foreach ($path in @(
        $featurePath,
        $proofPath,
        $pipelinePath,
        $archivePath
    )) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            return $false
        }
    }

    $archive = Get-Content -LiteralPath $archivePath -Raw | ConvertFrom-Json
    if ([int]$archive.fold_id -ne $Fold -or $archive.status -ne "cleaned") {
        return $false
    }
    $expected = Get-ExpectedFeatureHash -FoldDirectory $Directory
    $actual = (
        Get-FileHash -LiteralPath $featurePath -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    return $actual -eq $expected
}

function Test-RemoteFoldReady {
    param([int]$Fold)

    $remoteFold = (
        "$RemoteProject/knowledge/null_release_oof/roberta128/fold$Fold"
    )
    $command = @(
        "test -s '$remoteFold/heldout_features.pt'",
        "test -s '$remoteFold/heldout_features.pt.sha256'",
        "test -s '$remoteFold/fold_proof.json'",
        "test -s '$remoteFold/pipeline_manifest.json'",
        "test -s '$remoteFold/fold_archive_manifest.json'",
        "echo READY"
    ) -join " && "
    $result = & ssh $Server $command 2>$null
    return $LASTEXITCODE -eq 0 -and ($result -contains "READY")
}

function Pull-Fold {
    param([int]$Fold)

    $destination = Join-Path $LocalRoot "fold$Fold"
    if (Test-Path -LiteralPath $destination) {
        if (Test-LocalFold -Fold $Fold -Directory $destination) {
            Write-ProgressLog "Fold $Fold already exists locally and passed SHA-256."
            return $true
        }
        throw "Local fold $Fold exists but failed validation: $destination"
    }
    if (-not (Test-RemoteFoldReady -Fold $Fold)) {
        Write-ProgressLog "Fold $Fold is not sealed and cleaned on the server yet."
        return $false
    }

    $staging = Join-Path $LocalRoot ".fold${Fold}.pull-$PID"
    Assert-UnderLocalRoot -Path $staging
    if (Test-Path -LiteralPath $staging) {
        Remove-Item -LiteralPath $staging -Recurse -Force
    }

    $remoteFold = (
        "$RemoteProject/knowledge/null_release_oof/roberta128/fold$Fold"
    )
    Write-ProgressLog "Downloading fold $Fold from ${Server}:${remoteFold}."
    & scp -q -r "${Server}:${remoteFold}" $staging
    if ($LASTEXITCODE -ne 0) {
        throw "scp failed for fold $Fold with exit code $LASTEXITCODE."
    }
    if (-not (Test-LocalFold -Fold $Fold -Directory $staging)) {
        throw "Downloaded fold $Fold failed manifest or SHA-256 validation."
    }

    Move-Item -LiteralPath $staging -Destination $destination
    Write-ProgressLog "Fold $Fold stored and verified: $destination"
    return $true
}

do {
    $complete = 0
    foreach ($fold in $Folds) {
        if (Pull-Fold -Fold $fold) {
            $complete += 1
        }
    }
    if ($complete -eq $Folds.Count) {
        Write-ProgressLog "All requested folds are stored locally."
        break
    }
    if (-not $Watch) {
        Write-ProgressLog (
            "Stored $complete/$($Folds.Count) ready folds; rerun later for the rest."
        )
        break
    }
    Write-ProgressLog (
        "Stored $complete/$($Folds.Count); polling again in $PollSeconds seconds."
    )
    Start-Sleep -Seconds $PollSeconds
} while ($true)
