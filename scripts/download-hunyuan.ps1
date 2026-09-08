param(
  [string]$Repository = "tencent/Hunyuan3D-2mv"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$modelRoot = Join-Path $projectRoot "models\hunyuan3d-2mv"
New-Item -ItemType Directory -Force -Path $modelRoot | Out-Null

Write-Host "Hunyuan3D-2mv is distributed under Tencent Hunyuan Community License terms."
Write-Host "Confirm the model-page terms with the intended use before downloading."
Write-Host "Downloading only the non-turbo multiview fp16 safetensors checkpoint (about 5 GB)."
# Positional filenames are intentional: some hf CLI releases only honour the
# last repeated --include pattern.  --local-dir creates the layout consumed
# offline by HY3DGEN_MODELS.
hf download $Repository `
  "hunyuan3d-dit-v2-mv/config.yaml" `
  "hunyuan3d-dit-v2-mv/model.fp16.safetensors" `
  "LICENSE" `
  "NOTICE" `
  --local-dir $modelRoot
if ($LASTEXITCODE -ne 0) {
  throw "Hunyuan3D-2mv download failed. Confirm hf auth and the model-page access approval."
}

$config = Join-Path $modelRoot "hunyuan3d-dit-v2-mv\config.yaml"
$weights = Join-Path $modelRoot "hunyuan3d-dit-v2-mv\model.fp16.safetensors"
if (-not ((Test-Path $config) -and (Test-Path $weights))) {
  throw "The required config or safetensors file is missing; the incomplete download was not accepted."
}
Write-Host "Done. Docker mounts models\hunyuan3d-2mv read-only for offline generation."
