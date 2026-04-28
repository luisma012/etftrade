# AUDIT BOT — Reglas para pasar a cuenta REAL

## Estado actual: PAPER MODE
- Cuenta paper: ~$100,000 simulados
- P&L actual: +$1,603.39 (+5.45%)
- Fecha inicio paper: ~marzo 2025

---

## CHECKLIST OBLIGATORIO antes de pasar a LIVE

### FASE 1 — Paper Trading (mínimo 30 días)
- [ ] Bot corriendo en paper mínimo 30 días consecutivos
- [ ] Fecha inicio paper: ___________
- [ ] Fecha fin paper (mínimo): ___________
- [ ] Win rate > 55%
- [ ] Profit Factor > 1.3
- [ ] Drawdown máximo < 10%
- [ ] ROI positivo en el período
- [ ] Todas las alertas Telegram funcionando correctamente
- [ ] Stop-loss ejecutándose correctamente en todas las posiciones
- [ ] Breakeven activándose cuando sube 1x ATR
- [ ] Take-profit ejecutándose a 3x ATR
- [ ] Cooldown de 3 días respetándose tras stop-loss
- [ ] Filtro VIX bloqueando entradas cuando VIX > 20
- [ ] Posiciones sincronizándose correctamente tras reinicio del bot
- [ ] Sin errores críticos en logs por 7 días consecutivos

### FASE 2 — Backtest Validación
- [ ] Backtest SPY mínimo 1 año — Win rate: ___% — PF: ___
- [ ] Backtest VGT mínimo 1 año — Win rate: ___% — PF: ___
- [ ] Backtest SCHD mínimo 1 año — Win rate: ___% — PF: ___
- [ ] Backtest TSLA mínimo 1 año — Win rate: ___% — PF: ___
- [ ] Backtest en mercado bajista (ej. 2022) — resultado: ___________
- [ ] Drawdown máximo en backtest < 15%
- [ ] Profit Factor promedio > 1.2

### FASE 3 — Seguridad y Configuración
- [ ] API keys de cuenta LIVE generadas (diferentes a paper)
- [ ] API keys con permisos SOLO de trading (sin transferencias)
- [ ] .env NO está en git (.gitignore verificado)
- [ ] Servidor/PC estable con UPS o cloud (evitar cortes de luz)
- [ ] Telegram alertas verificadas (recibes notificaciones)
- [ ] cooldown.json respaldado

---

## PARÁMETROS RECOMENDADOS PARA LIVE

### Semana 1-2: Modo conservador
```
AUTO_CONFIDENCE       = 0.90   # subir de 85% a 90%
AUTO_MAX_POSITION_USD = 200    # empezar con $200 por trade
VIX_MAX               = 18     # más estricto con volatilidad
COOLDOWN_DAYS         = 5      # más días de enfriamiento
```

### Semana 3-4: Modo moderado
```
AUTO_CONFIDENCE       = 0.88
AUTO_MAX_POSITION_USD = 500
VIX_MAX               = 19
COOLDOWN_DAYS         = 4
```

### Mes 2+: Modo normal (si resultados positivos)
```
AUTO_CONFIDENCE       = 0.85
AUTO_MAX_POSITION_USD = 1000
VIX_MAX               = 20
COOLDOWN_DAYS         = 3
```

---

## MÉTRICAS MÍNIMAS PARA MANTENERSE EN LIVE

| Métrica | Mínimo | Acción si falla |
|---------|--------|-----------------|
| Win Rate | > 50% | Revisar modelo XGBoost |
| Profit Factor | > 1.0 | Pausar auto-trade, solo manual |
| Drawdown semanal | < 5% | Reducir posición a $200 |
| Drawdown mensual | < 10% | PAUSAR bot, volver a paper |
| Pérdidas consecutivas | < 5 | Aumentar cooldown a 7 días |
| VIX sostenido > 25 | N/A | Desactivar auto-trade |

---

## CÓMO PASAR A LIVE

### Paso 1: Obtener API keys de cuenta real
1. Ir a https://app.alpaca.markets
2. Crear cuenta real (requiere verificación de identidad)
3. Depositar fondos
4. Generar API Key y Secret Key para la cuenta LIVE

### Paso 2: Modificar .env
```
# ANTES (paper)
ALPACA_API_KEY=PKXAF5PXQEXAYOTLRSMTSMTPCD
ALPACA_SECRET_KEY=2AwfG9Z8pebFkoVcdM1PEuxGC3L2Hxi1uf7KBtUNnPec
ALPACA_PAPER=true

# DESPUÉS (live) — usar keys de cuenta REAL
ALPACA_API_KEY=TU_API_KEY_LIVE
ALPACA_SECRET_KEY=TU_SECRET_KEY_LIVE
ALPACA_PAPER=false
```

### Paso 3: Reiniciar bot
- Cerrar bot actual (Ctrl+C)
- Ejecutar run.bat
- Verificar que el dashboard muestre "LIVE 🔴"
- Verificar saldo real en el panel

### Paso 4: Monitoreo intensivo (primeras 48 horas)
- Revisar Telegram cada hora
- Verificar que stops se ejecutan
- NO alejarse del dashboard el primer día
- Tener acceso a Alpaca web para cerrar posiciones manualmente si hay error

---

## REGLAS DE EMERGENCIA

### PAUSAR INMEDIATAMENTE si:
1. Pérdida > 3% del portfolio en un solo día
2. Orden ejecutada que no esperabas
3. Error repetido en logs
4. WebSocket desconectado por > 5 minutos durante mercado abierto
5. VIX sube por encima de 30

### CÓMO PAUSAR:
- Opción 1: Toggle "Auto OFF" en el dashboard
- Opción 2: Cerrar el bot (Ctrl+C o cerrar consola)
- Opción 3: Cancelar órdenes desde https://app.alpaca.markets
- Opción 4: Cambiar .env a ALPACA_PAPER=true y reiniciar

### VOLVER A PAPER si:
- Drawdown mensual > 10%
- 5+ pérdidas consecutivas
- Bug descubierto en la lógica
- Cambio importante en el código

---

## RESUMEN DE RIESGOS

| Riesgo | Mitigación actual |
|--------|-------------------|
| Pérdida grande | Stop-loss ATR + max $1000/trade |
| Mercado en pánico | Filtro VIX > 20 bloquea |
| Sobreoperar | 1 entrada/símbolo/día + cooldown |
| Bot se cae | Posiciones sincronizadas al reiniciar |
| Signal falsa | Umbral 85% + 25 features técnicos |
| Flash crash | Órdenes tipo DAY (no overnight) |
| Corte de internet | Stops de Alpaca siguen activos server-side* |

*NOTA: Los stops del bot son SOFTWARE, no están en el servidor de Alpaca.
Si el bot se cae, NO hay stop-loss activo. Considerar implementar
stops server-side con Alpaca bracket orders para cuenta LIVE.

---

## LÍNEA DE TIEMPO RECOMENDADA

| Período | Acción | Duración |
|---------|--------|----------|
| Ahora | Seguir en paper, monitorear | 30 días mín |
| Día 30 | Revisar métricas vs este checklist | 1 día |
| Día 31 | Si todo OK → LIVE con $200/trade | 14 días |
| Día 45 | Si positivo → subir a $500/trade | 14 días |
| Día 60 | Si positivo → subir a $1000/trade | continuo |
| Mensual | Revisar métricas, ajustar | recurrente |

**TIEMPO TOTAL MÍNIMO ANTES DE LIVE COMPLETO: ~60 días**
