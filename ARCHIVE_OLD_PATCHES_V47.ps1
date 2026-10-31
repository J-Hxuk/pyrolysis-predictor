<#
Archives old patch ZIPs from the user's EXISTING project; never touches data, models, PDFs.
Dry-run is the default. Move only with -Apply after v47 tests pass.
#>
param(
    [Parameter(Mandatory=$true)][string]$SourceProjectRoot,
    [Parameter(Mandatory=$true)][string]$ArchiveRoot,
    [switch]$Apply
)
$ErrorActionPreference='Stop'
$source=(Resolve-Path -LiteralPath $SourceProjectRoot).Path
if (-not (Test-Path -LiteralPath $source -PathType Container)) {throw "Not a directory: $source"}
$archive=[System.IO.Path]::GetFullPath($ArchiveRoot)
if ($source.TrimEnd('\') -ieq $archive.TrimEnd('\')) {throw '원본과 보관 폴더가 같습니다.'}
# Only completed incremental patches/fixes; never archive the v47 baseline ZIP.
$pattern='^Biochar_(?:v\d+|Stage1_v\d+|Stage2_v\d+|Phase8_v\d+).*_(?:Patch(?:_SAFE)?|Fix)\.zip$'
$candidates=Get-ChildItem -LiteralPath $source -File | Where-Object {$_.Name -match $pattern}
if (-not $candidates) {Write-Host '이동 가능한 과거 패치 ZIP 없음';exit 0}
foreach ($file in $candidates) {
    $target=Join-Path $archive $file.Name
    if (Test-Path -LiteralPath $target) { Write-Host "SKIP: archive already has $($file.Name)"; continue }
    if ($Apply) {
        New-Item -ItemType Directory -Path $archive -Force | Out-Null
        Move-Item -LiteralPath $file.FullName -Destination $target -ErrorAction Stop
        Write-Host "MOVED: $($file.Name)"
    } else {Write-Host "PLAN: $($file.Name) -> $archive"}
}
if (-not $Apply) {Write-Host '미리보기 전용입니다. 파일 이동은 -Apply 명시 시에만 실행됩니다.'}
