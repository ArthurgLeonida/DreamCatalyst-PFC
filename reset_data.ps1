# Reset and reprocess chair data
Write-Host "Backing up original images..."
New-Item -ItemType Directory -Path "data\chair_backup" -Force | Out-Null
Get-ChildItem "data\chair\images" -File | Where-Object { $_.Name -like "IMG_*" } | Copy-Item -Destination "data\chair_backup" -Force
$backupCount = (Get-ChildItem "data\chair_backup" -File).Count
Write-Host "  Backed up $backupCount images"

Write-Host "Deleting processed data..."
Remove-Item "data\chair\colmap" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "data\chair\images" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "data\chair\images_2" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "data\chair\images_4" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "data\chair\images_8" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "data\chair\transforms.json" -Force -ErrorAction SilentlyContinue
Remove-Item "data\chair\sparse_pc.ply" -Force -ErrorAction SilentlyContinue
Write-Host "  Cleaned"

Write-Host "Restoring original images..."
New-Item -ItemType Directory -Path "data\chair\images" -Force | Out-Null
Move-Item "data\chair_backup\*" "data\chair\images\" -Force
Remove-Item "data\chair_backup" -Force
$imageCount = (Get-ChildItem "data\chair\images" -File).Count
Write-Host "  Restored $imageCount images"

Write-Host ""
Write-Host "Ready! Now run: ns-process-data images --data data/chair/images --output-dir data/chair"
