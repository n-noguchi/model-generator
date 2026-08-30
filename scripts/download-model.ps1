param(
  [string]$Repository = "stabilityai/stable-fast-3d"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$cacheRoot = Join-Path $projectRoot "models\huggingface"
$hubCache = Join-Path $cacheRoot "hub"
New-Item -ItemType Directory -Force -Path $hubCache | Out-Null

Write-Host "Stable Fast 3D is a gated model. Authenticate in a browser first: hf auth login"
Write-Host "Downloading $Repository, facebook/dinov2-large, and CLIP ViT-B/32 into $hubCache"
# The Hugging Face CLI obtains credentials from the host profile/environment; no token is read or saved here.
hf download $Repository --cache-dir $hubCache
if ($LASTEXITCODE -ne 0) { throw "Hugging Face download failed. Confirm access approval and run 'hf auth login'." }
hf download facebook/dinov2-large --cache-dir $hubCache
if ($LASTEXITCODE -ne 0) { throw "DINOv2 download failed. Confirm host connectivity and retry." }
hf download laion/CLIP-ViT-B-32-laion2B-s34B-b79K --cache-dir $hubCache
if ($LASTEXITCODE -ne 0) { throw "CLIP download failed. Confirm host connectivity and retry." }
Write-Host "Done. The cache can be used by docker compose with HF_HUB_OFFLINE=1."
