$py = 'C:\Users\HP\anaconda3\python.exe'
$script = 'C:\Users\HP\.agents\skills\backtesting-trading-strategies\scripts\backtest.py'

$etfs = @('SPY','VGT','SCHD','IFRA','SCHF')
$strategies = @('macd','sma_crossover','rsi_reversal','bollinger_bands')

foreach ($etf in $etfs) {
    foreach ($strat in $strategies) {
        Write-Host "=== $strat / $etf ==="
        $output = & $py $script --strategy $strat --symbol $etf --period 2y --capital 100000 --quiet 2>&1
        $output | ForEach-Object { Write-Host $_ }
        Write-Host ""
    }
}
