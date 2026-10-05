# ChainGuard End-to-End Demo Script (Windows / PowerShell)
# =========================================================

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "   ChainGuard Protocol: End-to-End Demonstration" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

# 1. Start Hardhat node in background if not already running
Write-Host "`n[Step 1] Starting Hardhat EVM Node..." -ForegroundColor Yellow
$hardhatProcess = Start-Process -FilePath "npx" -ArgumentList "hardhat node" -WorkingDirectory "contracts" -PassThru -NoNewWindow
Start-Sleep -Seconds 4

# 2. Deploy Contracts
Write-Host "`n[Step 2] Deploying ModelRegistry & RiskRegistry to Local Node..." -ForegroundColor Yellow
Set-Location contracts
npx hardhat run scripts/deploy.ts --network localhost
Set-Location ..

# 3. Publish Demo Batch On-Chain
Write-Host "`n[Step 3] Scoring Demo Transactions & Publishing Batch On-Chain..." -ForegroundColor Yellow
python service/publisher.py --sample-size 20

# 4. Start FastAPI Backend & Static DApp
Write-Host "`n[Step 4] Launching Services..." -ForegroundColor Yellow
Write-Host "  - Inference API: http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "  - DApp Frontend: http://127.0.0.1:3000" -ForegroundColor Green
Write-Host "`nTo start servers in separate terminals, run:"
Write-Host "  Terminal 1: uvicorn service.app.main:app --host 127.0.0.1 --port 8000"
Write-Host "  Terminal 2: cd dapp; npx serve -l 3000 ."

Write-Host "`nDemo Setup Completed Successfully!" -ForegroundColor Cyan
