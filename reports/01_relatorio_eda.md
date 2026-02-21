# Relatório de Análise Exploratória de Dados (EDA)
## Tech Challenge Fase 3 - Machine Learning Engineering

**Projeto:** Análise e Predição de Atrasos de Voos nos EUA  
**Dataset:** Voos domésticos nos Estados Unidos - Ano 2015  
**Data do Relatório:** Fevereiro de 2026

---

## Sumário Executivo

Este relatório apresenta os resultados da análise exploratória de dados (EDA) realizada sobre o dataset de voos domésticos nos Estados Unidos referente ao ano de 2015. O objetivo principal é compreender os padrões de atrasos de voos e identificar fatores que contribuem para esses atrasos, preparando o terreno para a construção de modelos preditivos.

### Principais Descobertas

| Métrica | Valor |
|---------|-------|
| Total de voos analisados | 5.819.079 |
| Taxa de atraso (≥15 min) | 18,56% |
| Taxa de cancelamento | 1,54% |
| Companhias aéreas | 14 |
| Aeroportos | 628-629 |
| Rotas únicas | 8.609 |

---

## 1. Estrutura e Qualidade dos Dados

### 1.1 Visão Geral do Dataset

O dataset original contém **5.819.079 registros** (voos) e **31 colunas**, ocupando aproximadamente **1,9 GB** de memória. Após o enriquecimento com informações de companhias aéreas e aeroportos, o dataset expandiu para **50 colunas**.

#### Período dos Dados
- **Início:** 01/01/2015
- **Fim:** 31/12/2015
- **Cobertura:** 1 ano completo de operações

### 1.2 Análise de Valores Nulos

A análise de valores nulos revelou padrões importantes que refletem a natureza dos dados:

| Coluna | Nulos | Percentual | Interpretação |
|--------|-------|------------|---------------|
| CANCELLATION_REASON | 5.729.195 | 98,46% | Esperado: só preenchido para voos cancelados |
| Colunas de motivo de atraso* | 4.755.640 | 81,72% | Só preenchidas quando há atraso significativo |
| ARRIVAL_DELAY | 105.071 | 1,81% | Voos cancelados ou desviados |
| DEPARTURE_TIME | 86.153 | 1,48% | Corresponde aos voos cancelados |
| TAIL_NUMBER | 14.721 | 0,25% | Dados não registrados |

*AIR_SYSTEM_DELAY, SECURITY_DELAY, AIRLINE_DELAY, LATE_AIRCRAFT_DELAY, WEATHER_DELAY

#### Contexto dos Valores Nulos

Os valores nulos têm explicações lógicas:

- **Voos Cancelados (89.884 - 1,54%):** Não possuem horários de partida/chegada reais, o que explica os nulos em DEPARTURE_TIME, ARRIVAL_DELAY, etc.
- **Voos Desviados (15.187 - 0,26%):** Podem ter dados incompletos de chegada.
- **Colunas de Motivo de Atraso:** Só são preenchidas quando há atraso ≥15 minutos, por isso 81,72% são nulos (apenas 18,28% dos voos tiveram atraso registrado com detalhamento de causa).

### 1.3 Estatísticas Descritivas das Variáveis Numéricas

| Variável | Média | Mediana | Desvio Padrão | Mín | Máx |
|----------|-------|---------|---------------|-----|-----|
| DEPARTURE_DELAY | 9,37 min | -2,00 min | 37,08 | -82 | 1.988 |
| ARRIVAL_DELAY | 4,41 min | -5,00 min | 39,27 | -87 | 1.971 |
| DISTANCE | 822 mi | 647 mi | 607,78 | 21 | 4.983 |
| SCHEDULED_TIME | 141,69 min | 123 min | 75,21 | 18 | 718 |
| TAXI_OUT | 16,07 min | 14 min | 8,90 | 1 | 225 |
| TAXI_IN | 7,43 min | 6 min | 5,64 | 1 | 248 |

**Observações importantes:**
- A mediana do ARRIVAL_DELAY é **-5 minutos**, indicando que mais da metade dos voos chegam adiantados.
- Existem outliers extremos de atraso (até ~33 horas de atraso máximo).
- A distância média dos voos é de 822 milhas (~1.323 km).

---

## 2. Análise de Atrasos

### 2.1 Distribuição Geral dos Atrasos

Considerando apenas os **5.729.195 voos válidos** (não cancelados):

| Categoria de Atraso | Descrição | Observações |
|---------------------|-----------|-------------|
| Adiantado | Chegou antes do previsto | Maioria dos voos |
| No Horário | Chegou no horário exato | Minoria |
| Atraso Leve | < 15 minutos | Não considerado "atraso oficial" |
| Atraso Moderado | 15-60 minutos | ~15% dos voos |
| Atraso Significativo | 1-2 horas | ~2,5% dos voos |
| Atraso Severo | > 2 horas | ~0,5% dos voos |

**Taxa oficial de atraso (≥15 min):** 18,56% (1.063.439 voos)

### 2.2 Atrasos por Companhia Aérea

#### Ranking de Companhias por Taxa de Atraso

| Posição | Companhia | Taxa de Atraso | Atraso Médio | Total de Voos |
|---------|-----------|----------------|--------------|---------------|
| 1 (pior) | Spirit Air Lines | 30% | 14,47 min | 115.375 |
| 2 | Frontier Airlines | 26% | 12,50 min | 90.248 |
| 3 | JetBlue Airways | 23% | 6,68 min | 262.772 |
| 4 | American Eagle Airlines | 22% | 6,46 min | 279.607 |
| 5 | United Air Lines | 21% | 5,43 min | 509.150 |
| ... | ... | ... | ... | ... |
| 13 | Alaska Airlines | 13% | -0,98 min | 171.852 |
| 14 (melhor) | Hawaiian Airlines | 11% | 2,02 min | 76.101 |

**Insights:**
- **Spirit Airlines** lidera com 30% de atrasos, quase 3x mais que Hawaiian Airlines.
- **Hawaiian Airlines** tem a menor taxa (11%) e consistentemente bom desempenho.
- **Alaska Airlines** é a única companhia com atraso médio negativo (-0,98 min), ou seja, em média chega adiantada.
- As companhias low-cost (Spirit, Frontier) tendem a ter maiores taxas de atraso.

### 2.3 Aeroportos Mais Problemáticos

#### Top 10 Aeroportos de Origem com Maior Taxa de Atraso

| Pos | Aeroporto | Cidade | Taxa | Atraso Médio | Voos |
|-----|-----------|--------|------|--------------|------|
| 1 | HPN | White Plains, NY | 25% | 9,6 min | 7.164 |
| 2 | LGA | Nova York (LaGuardia) | 24% | 5,6 min | 95.074 |
| 3 | ORD | Chicago (O'Hare) | 24% | 8,6 min | 277.336 |
| 4 | BWI | Baltimore | 23% | 7,2 min | 84.546 |
| 5 | XNA | Fayetteville, AR | 23% | 11,2 min | 8.987 |
| 6 | MIA | Miami | 23% | 7,0 min | 68.558 |
| 7 | HOU | Houston (Hobby) | 22% | 7,6 min | 51.268 |
| 8 | BTR | Baton Rouge | 22% | 11,2 min | 7.003 |
| 9 | DEN | Denver | 22% | 7,2 min | 193.932 |
| 10 | MDW | Chicago (Midway) | 22% | 6,0 min | 78.927 |

#### Top 10 Aeroportos de Destino com Maior Taxa de Atraso

| Pos | Aeroporto | Taxa | Atraso Médio | Voos |
|-----|-----------|------|--------------|------|
| 1 | LGA | 26% | 10,4 min | 95.163 |
| 2 | COS | 24% | 10,2 min | 6.771 |
| 3 | TYS | 24% | 10,6 min | 6.774 |
| 4 | HPN | 24% | 9,0 min | 7.207 |
| 5 | LAX | 23% | 6,1 min | 192.437 |

**Insights:**
- **LaGuardia (LGA)** aparece como problemático tanto na origem quanto no destino.
- **Chicago O'Hare (ORD)** é um dos maiores hubs e tem alta taxa de atraso (24%).
- Aeroportos do Nordeste dos EUA (NY, NJ, MA) têm taxas consistentemente altas.

### 2.4 Estados com Maior Taxa de Atraso

| Estado | Taxa de Atraso | Atraso Médio | Volume de Voos |
|--------|----------------|--------------|----------------|
| Illinois (IL) | 23,4% | 8,0 min | 370.549 |
| Maryland (MD) | 22,5% | 7,2 min | 84.546 |
| Colorado (CO) | 22,1% | 7,2 min | 212.735 |
| New York (NY) | 21,7% | 4,1 min | 238.531 |
| New Jersey (NJ) | 21,5% | 4,8 min | 104.973 |
| Texas (TX) | 20,5% | 6,7 min | 618.298 |

**Insight:** Estados do Centro-Norte e Nordeste dos EUA concentram as maiores taxas de atraso, possivelmente devido ao clima severo no inverno e à alta densidade de tráfego aéreo.

---

## 3. Análise Temporal

### 3.1 Padrões por Hora do Dia

#### Horários Mais Críticos (Maior Taxa de Atraso)

| Hora | Taxa de Atraso | Atraso Médio | Volume |
|------|----------------|--------------|--------|
| 20h | 26,3% | 9,8 min | 254.179 |
| 19h | 26,2% | 10,1 min | 325.209 |
| 18h | 25,8% | 9,9 min | 327.884 |
| 17h | 24,4% | 9,0 min | 383.576 |
| 21h | 24,3% | 8,1 min | 183.786 |

#### Melhores Horários (Menor Taxa de Atraso)

| Hora | Taxa de Atraso | Atraso Médio | Volume |
|------|----------------|--------------|--------|
| 05h | 7,2% | -3,8 min | 115.984 |
| 06h | 9,0% | -2,6 min | 399.618 |
| 07h | 11,2% | -1,5 min | 388.704 |
| 08h | 12,9% | -0,2 min | 375.689 |
| 09h | 14,2% | 1,0 min | 347.064 |

**Insight crítico:** 
- **Voos matutinos (5h-9h)** têm taxa de atraso **3-4x menor** que voos noturnos.
- O "efeito cascata" acumula atrasos ao longo do dia.
- **Recomendação:** Para evitar atrasos, priorizar voos no período da manhã.

### 3.2 Padrões por Dia da Semana

O maior atraso médio registrado foi de **13,0 minutos** às **quintas-feiras às 19h**, enquanto o menor foi **-4,7 minutos** (adiantado) às **quintas-feiras às 5h**.

### 3.3 Sazonalidade Mensal por Companhia

| Companhia | Pior Mês | Atraso | Melhor Mês | Atraso |
|-----------|----------|--------|------------|--------|
| Spirit (NK) | Junho | 35,6 min | Outubro | 6,8 min |
| Frontier (F9) | Fevereiro | 27,4 min | Outubro | 0,1 min |
| American Eagle (MQ) | Fevereiro | 21,3 min | Outubro | -3,7 min |
| JetBlue (B6) | Fevereiro | 18,7 min | Maio | -0,7 min |
| United (UA) | Junho | 16,9 min | Outubro | -2,5 min |

**Insight:** 
- **Fevereiro** é consistentemente ruim (tempestades de inverno).
- **Junho** afeta principalmente companhias do centro-oeste (temporada de tempestades).
- **Setembro/Outubro** são os melhores meses para viajar.

### 3.4 Dias com Picos de Atraso

| Data | Atraso Médio | Taxa de Atraso | Voos |
|------|--------------|----------------|------|
| 04/01/2015 | 32,0 min | 49,2% | 15.919 |
| 27/12/2015 | 28,2 min | 39,4% | 15.351 |
| 29/12/2015 | 26,3 min | 39,8% | 15.514 |
| 30/12/2015 | 26,2 min | 43,5% | 15.979 |
| 01/03/2015 | 25,9 min | 40,9% | 13.657 |

**Insight:** Os piores dias coincidem com:
- Período pós-festas de fim de ano (27-30 dezembro)
- Tempestade de inverno em janeiro
- Início de março (transição de estação)

---

## 4. Rotas Mais Problemáticas

### Top 10 Rotas com Maior Taxa de Atraso

| Rota | Taxa | Atraso Médio | Voos |
|------|------|--------------|------|
| DFW → HNL (Dallas → Honolulu) | 40% | 28,6 min | 695 |
| ORD → ASE (Chicago → Aspen) | 39% | 22,3 min | 597 |
| LGA → MYR (NY → Myrtle Beach) | 38% | 17,8 min | 619 |
| ORD → BOI (Chicago → Boise) | 37% | 18,5 min | 527 |
| DCA → JFK (Washington → NY) | 36% | 21,0 min | 882 |
| ORD → COS (Chicago → Colorado Springs) | 35% | 18,6 min | 911 |
| LGA → CMH (NY → Columbus) | 34% | 12,4 min | 1.316 |
| IAH → LAX (Houston → Los Angeles) | 33% | 16,7 min | 4.393 |
| RDU → LGA (Raleigh → NY) | 33% | 17,2 min | 1.681 |
| RIC → LGA (Richmond → NY) | 33% | 17,9 min | 1.287 |

**Padrões identificados:**
- **Chicago O'Hare (ORD)** aparece em 3 das 10 piores rotas.
- **LaGuardia (LGA)** aparece em 4 das 10 piores rotas (origem ou destino).
- Rotas longas (transcontinentais) como DFW→HNL têm atrasos mais severos.
- Rotas para destinos turísticos de inverno (Aspen) sofrem com condições climáticas.

---

## 5. Análise de Causas de Atraso

### 5.1 Composição dos Tipos de Atraso

| Tipo de Atraso | Total de Minutos | Voos Afetados | Média quando Ocorre |
|----------------|------------------|---------------|---------------------|
| Aeronave Atrasada | 24.961.931 | 556.953 | 44,8 min |
| Companhia Aérea | 20.172.956 | 570.022 | 35,4 min |
| Sistema Aéreo | 14.335.762 | 564.826 | 25,4 min |
| Clima | 3.100.233 | 64.716 | 47,9 min |
| Segurança | 80.985 | 3.484 | 23,2 min |

**Insights:**
- **"Aeronave Atrasada"** (Late Aircraft Delay) é a maior causa, representando o efeito cascata de atrasos.
- **Clima** afeta menos voos, mas quando ocorre, causa atrasos severos (média de 47,9 min).
- **Segurança** é raramente a causa (apenas 3.484 voos afetados no ano).

### 5.2 Principal Causa de Atraso por Companhia

| Companhia | Principal Causa | Média (min) |
|-----------|-----------------|-------------|
| Spirit (NK) | Sistema Aéreo | 27,51 |
| Hawaiian (HA) | Companhia Aérea | 22,79 |
| Delta (DL) | Companhia Aérea | 22,94 |
| Southwest (WN) | Aeronave Atrasada | 26,68 |
| United (UA) | Aeronave Atrasada | 26,02 |
| American (AA) | Aeronave Atrasada | 21,75 |
| Alaska (AS) | Aeronave Atrasada | 17,06 |

**Insight:** A maioria das companhias sofre principalmente com "Aeronave Atrasada", indicando problemas operacionais em cadeia. Apenas Spirit sofre mais com problemas do Sistema Aéreo.

### 5.3 Sazonalidade das Causas de Atraso

| Causa | Pior Mês | Média | Melhor Mês | Média |
|-------|----------|-------|------------|-------|
| Aeronave Atrasada | Junho | 26,40 min | Setembro | 19,73 min |
| Companhia Aérea | Outubro | 20,07 min | Janeiro | 17,80 min |
| Sistema Aéreo | Fevereiro | 14,18 min | Julho | 12,48 min |
| Clima | Fevereiro | 4,32 min | Outubro | 1,82 min |
| Segurança | Agosto | 0,13 min | Abril | 0,04 min |

**Insights:**
- **Fevereiro** tem pico de atrasos por clima (tempestades de inverno).
- **Junho** tem pico de aeronaves atrasadas (alta temporada + tempestades de verão).
- **Setembro/Outubro** são os meses mais tranquilos operacionalmente.

### 5.4 Análise dos Top 10 Aeroportos

| Aeroporto | Cidade | Atraso Total Médio | Principal Causa |
|-----------|--------|--------------------|-----------------| 
| ORD | Chicago | 62,7 min | Aeronave Atrasada |
| SFO | San Francisco | 58,7 min | Aeronave Atrasada |
| IAH | Houston | 58,2 min | Aeronave Atrasada |
| DFW | Dallas-Fort Worth | 57,6 min | Companhia Aérea |
| ATL | Atlanta | 56,8 min | Companhia Aérea |
| DEN | Denver | 55,8 min | Aeronave Atrasada |
| LAS | Las Vegas | 55,7 min | Aeronave Atrasada |
| MSP | Minneapolis | 55,4 min | Companhia Aérea |
| LAX | Los Angeles | 54,1 min | Aeronave Atrasada |
| PHX | Phoenix | 49,9 min | Aeronave Atrasada |

---

## 6. Análise de Cancelamentos

### 6.1 Visão Geral

- **Total de cancelamentos:** 89.884 voos (1,54% do total)
- **Total de desvios:** 15.187 voos (0,26% do total)

### 6.2 Motivos de Cancelamento

| Motivo | Quantidade | Percentual |
|--------|------------|------------|
| Clima | 48.851 | 54,3% |
| Companhia Aérea | 25.262 | 28,1% |
| Sistema Aéreo Nacional | 15.749 | 17,5% |
| Segurança | 22 | 0,0% |

**Insight:** O clima é responsável por mais da metade dos cancelamentos, algo que não pode ser controlado pelas companhias.

### 6.3 Sazonalidade dos Cancelamentos

| Mês | Taxa de Cancelamento | Voos Cancelados |
|-----|----------------------|-----------------|
| Fevereiro | 4,78% | 20.517 |
| Janeiro | 2,55% | 11.982 |
| Março | 2,18% | 11.002 |

**Insight:** O inverno (janeiro-março) concentra a maior parte dos cancelamentos, principalmente devido a tempestades de neve.

---

## 7. Matriz de Correlação

### Correlações Mais Fortes com ARRIVAL_DELAY

| Variável | Correlação |
|----------|------------|
| DEPARTURE_DELAY | 0,945 |
| AIRLINE_DELAY | 0,609 |
| LATE_AIRCRAFT_DELAY | 0,522 |
| WEATHER_DELAY | 0,265 |
| AIR_SYSTEM_DELAY | 0,247 |
| TAXI_OUT | 0,227 |

**Insights:**
- **DEPARTURE_DELAY** tem correlação quase perfeita (0,945) com ARRIVAL_DELAY - voos que saem atrasados quase sempre chegam atrasados.
- As causas específicas de atraso (AIRLINE_DELAY, LATE_AIRCRAFT_DELAY) têm correlação moderada.
- **DISTANCE** não aparece entre as correlações fortes, indicando que a distância do voo não é determinante para atrasos.

---

## 8. Variáveis Criadas para Modelagem

Durante a análise, foram criadas as seguintes variáveis derivadas:

| Variável | Descrição | Uso |
|----------|-----------|-----|
| IS_DELAYED | Indicador binário (1 se atraso ≥ 15 min) | Variável alvo para classificação |
| DELAY_CATEGORY | Categoria do atraso (6 níveis) | Análise e visualização |
| PERIOD | Período do dia (Manhã/Tarde/Noite/Madrugada) | Feature para modelagem |
| HOUR | Hora do dia (0-23) | Feature para modelagem |
| DATE | Data completa | Análise temporal |
| MONTH_NAME | Nome do mês | Visualização |
| DAY_NAME | Nome do dia da semana | Visualização |
| ROUTE | Rota (Origem → Destino) | Análise de rotas |
| DISTANCE_CATEGORY | Categoria de distância (4 níveis) | Feature para modelagem |

---

## 9. Conclusões e Recomendações

### 9.1 Principais Conclusões

1. **Taxa de Atraso Significativa:** 18,56% dos voos têm atraso ≥ 15 minutos, afetando mais de 1 milhão de passageiros.

2. **Efeito Cascata:** A principal causa de atraso é "Aeronave Atrasada", evidenciando que atrasos se propagam ao longo do dia.

3. **Padrão Temporal Claro:** 
   - Voos matutinos (5h-9h) têm 3-4x menos atrasos que voos noturnos.
   - Setembro/Outubro são os melhores meses; Fevereiro/Junho os piores.

4. **Aeroportos Problemáticos:** LaGuardia (LGA) e Chicago O'Hare (ORD) são consistentemente problemáticos.

5. **Companhias com Melhor Desempenho:** Hawaiian Airlines e Alaska Airlines têm as menores taxas de atraso.

6. **Clima como Fator Crítico:** Responsável por 54% dos cancelamentos e picos de atraso em fevereiro.

### 9.2 Recomendações para Modelagem

#### Features Candidatas para Modelos Preditivos:

**Alta Importância:**
- SCHEDULED_DEPARTURE (hora do voo)
- MONTH (sazonalidade)
- DAY_OF_WEEK
- AIRLINE
- ORIGIN_AIRPORT
- DESTINATION_AIRPORT

**Média Importância:**
- DISTANCE / DISTANCE_CATEGORY
- PERIOD (período do dia)

**Features Derivadas Potenciais:**
- Histórico de atrasos da rota
- Condições climáticas (dados externos)
- Indicador de feriado/alta temporada
- Taxa histórica de atraso do aeroporto/companhia

### 9.3 Próximos Passos

1. **Modelagem Supervisionada:**
   - Classificação: Prever se um voo vai atrasar (IS_DELAYED)
   - Regressão: Prever o tempo de atraso (ARRIVAL_DELAY)

2. **Modelagem Não Supervisionada:**
   - Clusterização de aeroportos/rotas por perfil de atraso
   - PCA para redução de dimensionalidade

3. **Melhorias Potenciais:**
   - Incorporar dados climáticos
   - Criar features de atraso histórico
   - Considerar conexões/escalonamento de voos

---

## Anexos

### A. Datasets Gerados

| Arquivo | Descrição |
|---------|-----------|
| `flights_processed.parquet` | Dataset de voos válidos (não cancelados) com features |
| `flights_complete.parquet` | Dataset completo com todas as features |
| `airline_stats.csv` | Estatísticas agregadas por companhia |
| `airport_origin_stats.csv` | Estatísticas por aeroporto de origem |
| `airport_dest_stats.csv` | Estatísticas por aeroporto de destino |

### B. Figuras Geradas

1. `fig_01_distribuicao_atrasos.png` - Distribuição dos atrasos
2. `fig_02_top_aeroportos.png` - Top aeroportos por volume
3. `fig_03_distribuicao_temporal.png` - Distribuição temporal
4. `fig_04_distribuicao_distancia.png` - Distribuição de distância
5. `fig_05_atraso_companhia.png` - Atraso por companhia
6. `fig_06_atraso_distancia.png` - Atraso vs distância
7. `fig_07_matriz_correlacao.png` - Matriz de correlação
8. `fig_08_heatmap_dia_hora.png` - Heatmap dia x hora
9. `fig_09_heatmap_mes_airline.png` - Heatmap mês x companhia
10. `fig_10_composicao_atrasos.png` - Composição dos tipos de atraso
11. `fig_11_atraso_tipo_companhia.png` - Tipos de atraso por companhia
12. `fig_12_cancelamentos.png` - Análise de cancelamentos
13. `fig_13_cancelamentos_sazonalidade.png` - Sazonalidade dos cancelamentos

---

*Relatório gerado como parte do Tech Challenge Fase 3 - Machine Learning Engineering - FIAP*
