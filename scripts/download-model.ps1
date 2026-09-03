param(
  [string]$Repository = "stabilityai/stable-fast-3d"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$cacheRoot = Join-Path $projectRoot "models\huggingface"
$hubCache = Join-Path $cacheRoot "hub"
$rembgRoot = Join-Path $projectRoot "models\rembg"
New-Item -ItemType Directory -Force -Path $hubCache | Out-Null
New-Item -ItemType Directory -Force -Path $rembgRoot | Out-Null

Write-Host "Stable Fast 3D is a gated model. Authenticate in a browser first: hf auth login"
Write-Host "Downloading $Repository, facebook/dinov2-large, and CLIP ViT-B/32 into $hubCache"
# The Hugging Face CLI obtains credentials from the host profile/environment; no token is read or saved here.
hf download $Repository --cache-dir $hubCache
if ($LASTEXITCODE -ne 0) { throw "Hugging Face download failed. Confirm access approval and run 'hf auth login'." }
hf download facebook/dinov2-large --cache-dir $hubCache
if ($LASTEXITCODE -ne 0) { throw "DINOv2 download failed. Confirm host connectivity and retry." }
hf download laion/CLIP-ViT-B-32-laion2B-s34B-b79K --cache-dir $hubCache
if ($LASTEXITCODE -ne 0) { throw "CLIP download failed. Confirm host connectivity and retry." }
$rembgModel = Join-Path $rembgRoot "u2net_human_seg.onnx"
$rembgExpectedSha256 = "01EB6A29A5C4D8EDB30B56ADAD9BB3A2A0535338E480724A213E0ACFD2D1C73C"
$rembgValid = (Test-Path $rembgModel) -and ((Get-FileHash -LiteralPath $rembgModel -Algorithm SHA256).Hash -eq $rembgExpectedSha256)
if ($rembgValid) {
  Write-Host "rembg person-cutout model already exists: $rembgModel"
} else {
  Write-Host "Downloading official rembg u2net_human_seg model into $rembgModel"
  $rembgPartial = "$rembgModel.part"
  curl.exe --fail --location --output $rembgPartial "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx"
  if ($LASTEXITCODE -ne 0) {
    Remove-Item -LiteralPath $rembgPartial -Force -ErrorAction SilentlyContinue
    throw "rembg model download failed. Confirm network access and retry."
  }
  if ((Get-FileHash -LiteralPath $rembgPartial -Algorithm SHA256).Hash -ne $rembgExpectedSha256) {
    Remove-Item -LiteralPath $rembgPartial -Force
    throw "rembg model SHA-256 verification failed; the download was not installed."
  }
  Move-Item -LiteralPath $rembgPartial -Destination $rembgModel -Force
}
Write-Host "Done. The cache can be used by docker compose with HF_HUB_OFFLINE=1."
