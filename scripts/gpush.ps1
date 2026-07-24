param(
    [string]$ProxyHost = "127.0.0.1",
    [int]$ProxyPort = 12334,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PushArgs
)

function Resolve-GitPath {
    $candidates = @(
        "git",
        "C:\Program Files\Git\cmd\git.exe",
        "C:\Program Files\Git\bin\git.exe",
        "$env:LOCALAPPDATA\Programs\Git\cmd\git.exe",
        "$env:LOCALAPPDATA\Programs\Git\bin\git.exe"
    )

    foreach ($candidate in $candidates) {
        if ($candidate -eq "git") {
            $cmd = Get-Command git -ErrorAction SilentlyContinue
            if ($cmd) {
                return $cmd.Source
            }
            continue
        }

        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw "Git executable was not found."
}

function Set-GitProxy {
    param(
        [string]$GitExe,
        [string]$ProxyHostName,
        [int]$Port
    )

    $proxyUrl = "http://$ProxyHostName`:$Port"
    & $GitExe config --global http.proxy $proxyUrl
    if ($LASTEXITCODE -ne 0) { throw "Failed to set http.proxy" }

    & $GitExe config --global https.proxy $proxyUrl
    if ($LASTEXITCODE -ne 0) { throw "Failed to set https.proxy" }

    Write-Host "Proxy enabled: $proxyUrl"
}

function Unset-GitProxy {
    param([string]$GitExe)

    & $GitExe config --global --unset http.proxy 2>$null
    & $GitExe config --global --unset https.proxy 2>$null
    Write-Host "Proxy disabled for git"
}

function Test-ProxyPort {
    param(
        [string]$ProxyHostName,
        [int]$Port
    )

    try {
        return Test-NetConnection $ProxyHostName -Port $Port -InformationLevel Quiet -WarningAction SilentlyContinue
    }
    catch {
        return $false
    }
}

try {
    $gitExe = Resolve-GitPath

    if (Test-ProxyPort -ProxyHostName $ProxyHost -Port $ProxyPort) {
        Set-GitProxy -GitExe $gitExe -ProxyHostName $ProxyHost -Port $ProxyPort
    }
    else {
        Unset-GitProxy -GitExe $gitExe
    }

    if (-not $PushArgs -or $PushArgs.Count -eq 0) {
        $PushArgs = @("push")
    }
    else {
        $PushArgs = @("push") + $PushArgs
    }

    & $gitExe @PushArgs
    exit $LASTEXITCODE
}
catch {
    Write-Error $_
    exit 1
}
