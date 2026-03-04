# PowerShell script to test the MultiEndpointTox API
# Run this after starting the API with: uvicorn src.api.app:app --reload

$baseUrl = "http://localhost:8000"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MultiEndpointTox API Test Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Health check
Write-Host "1. Health Check:" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
    Write-Host "   Status: $($response.status)" -ForegroundColor Green
    Write-Host "   Models loaded: $($response.models_loaded)"
    Write-Host "   Endpoints: $($response.available_endpoints -join ', ')"
} catch {
    Write-Host "   Error: API not running. Start with: uvicorn src.api.app:app --reload" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Test 2: Docking status
Write-Host "2. Docking Status:" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/docking/status" -Method Get
    Write-Host "   Available: $($response.available)"
    if ($response.dependencies) {
        Write-Host "   Dependencies:"
        Write-Host "     - RDKit: $($response.dependencies.rdkit)"
        Write-Host "     - Vina CLI: $($response.dependencies.vina_cli)"
        Write-Host "     - Vina Python: $($response.dependencies.vina_python)"
    }
    if (-not $response.available -and $response.install_instructions) {
        Write-Host ""
        Write-Host "   Installation needed:" -ForegroundColor Yellow
        Write-Host $response.install_instructions
    }
} catch {
    Write-Host "   Error checking docking status: $_" -ForegroundColor Red
}
Write-Host ""

# Test 3: Single prediction
Write-Host "3. Single Prediction (Hepatotox):" -ForegroundColor Yellow
$body = @{
    smiles = "CC(=O)Nc1ccc(O)cc1"
    endpoint = "hepatotox"
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/predict" -Method Post -Body $body -ContentType "application/json"
    Write-Host "   SMILES: $($response.smiles)"
    Write-Host "   Prediction: $($response.prediction)"
    Write-Host "   Label: $($response.label)"
    Write-Host "   Probability: $([math]::Round($response.probability, 3))"
} catch {
    Write-Host "   Error: $_" -ForegroundColor Red
}
Write-Host ""

# Test 4: Multi-endpoint prediction
Write-Host "4. Multi-Endpoint Prediction:" -ForegroundColor Yellow
$body = @{
    smiles = "CC(=O)Nc1ccc(O)cc1"
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/predict/multi" -Method Post -Body $body -ContentType "application/json"
    Write-Host "   SMILES: $($response.smiles)"
    Write-Host "   Results:"
    foreach ($endpoint in $response.predictions.PSObject.Properties) {
        $pred = $endpoint.Value
        if ($pred.label) {
            Write-Host "     - $($endpoint.Name): $($pred.label) (p=$([math]::Round($pred.probability, 2)))"
        } elseif ($pred.prediction) {
            Write-Host "     - $($endpoint.Name): $([math]::Round($pred.prediction, 2))"
        }
    }
} catch {
    Write-Host "   Error: $_" -ForegroundColor Red
}
Write-Host ""

# Test 5: Ensemble prediction (if docking available)
Write-Host "5. Ensemble Prediction (ML + Docking):" -ForegroundColor Yellow
$body = @{
    smiles = "CC(=O)Nc1ccc(O)cc1"
    endpoint = "herg"
    include_docking = $true
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/predict/ensemble" -Method Post -Body $body -ContentType "application/json"
    Write-Host "   ML Prediction: $($response.ml_prediction.prediction)"
    if ($response.docking -and $response.docking.herg) {
        Write-Host "   Docking Affinity: $($response.docking.herg.affinity) kcal/mol"
    } else {
        Write-Host "   Docking: Not available (install Vina)" -ForegroundColor Yellow
    }
    if ($response.ensemble) {
        Write-Host "   Ensemble Score: $($response.ensemble.ensemble_score)"
        Write-Host "   Risk Level: $($response.ensemble.risk_level)"
    }
} catch {
    Write-Host "   Error: $_" -ForegroundColor Red
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
