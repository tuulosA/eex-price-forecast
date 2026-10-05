# Results explained

This page walks through the figures that show **how well the forecast works** and **what it bases
its predictions on**. It assumes no background in machine learning or electricity markets.

The project forecasts the German electricity price for each hour of the coming days. It does this in
two steps. First, three models forecast how much electricity German **wind** turbines and **solar**
panels will produce and how much electricity the country will use (the **load**). Then a fourth
model uses those three forecasts, together with weather and market information, to forecast the
**price**.

## A few terms first

| Term | Meaning |
|---|---|
| **Day-ahead price** | Electricity for each hour of tomorrow is traded in an auction today. Its result, in euros per megawatt-hour (**EUR/MWh**), is the price this project forecasts. |
| **MW / GW** | Megawatt / gigawatt: a rate of electricity production or use. 1 GW = 1,000 MW. Germany uses roughly 35-80 GW. |
| **Negative prices** | When far more power is produced than needed (typically sunny, windy holidays), producers can end up paying to deliver, and the price drops below zero. |
| **D+1** | The next delivery day: the day immediately after the forecast is made. |
| **MAE** | Mean absolute error: the average size of the forecast's miss, ignoring whether it was too high or too low. Lower is better. An MAE of 25 EUR/MWh means the forecast is off by 25 EUR/MWh in a typical hour. |
| **Holdout days** | 18 days from January to September 2026 that were kept aside and never used to tune or choose anything in the model. Scores on them are an honest test, like an exam the model never saw. |

## Part 1 - How good are the forecasts?

Each figure in this part shows the 18 holdout days, one small chart per day. For each day the model
was trained only on data from *before* that day, then asked to forecast it - just as it would be in
real use.

**How to read these charts.** The **black line** is what actually happened; the **coloured line** is
the forecast. The horizontal axis is the hour of the day (0 = midnight, 12 = noon, German time). The
title gives the date and that day's MAE. Each chart has **its own vertical scale**, because days
differ enormously: a calm winter day might vary by 60 EUR/MWh, a spring holiday by 600. So compare
the *shapes* between charts, not the heights.

### Price

![Actual vs forecast price on each holdout day](data/evaluation/price_eval_days_holdout.png)

Over the 18 days the price forecast is off by **26.15 EUR/MWh** in a typical hour.

- **The daily rhythm is usually right.** Prices are typically higher in the morning and evening,
  when people use more power, and lower around midday, when solar panels produce most. The forecast
  follows that pattern on most days, such as 13 January, 3 February, and 6 June.
- **Sharp evening spikes are underestimated.** On 4 March, 24 June, and 21 September the forecast
  rises at the right time but only about half as high as the real price.
- **Extreme negative prices are missed.** On Sunday 26 April and on the 1 May holiday, a flood of
  solar power pushed the price to -414 and -499 EUR/MWh. The forecast only dipped to about -90. In
  training, the most extreme 0.1% of prices at each end are deliberately capped so that a few freak
  hours do not distort the model, which also means it practically cannot reach such values.
- **Those three extreme days carry 42% of the total error.** Without them the MAE would be 18.03
  EUR/MWh. The sample also happens to contain more extreme days than a typical stretch of the year.
- **New Year's Day sits at the wrong level.** The real price stayed near zero most of the day; the
  forecast stayed 30-60 EUR/MWh higher.

### Wind generation

![Wind: actual vs forecast on each holdout day](data/evaluation/wind_eval_days_holdout.png)

Typical miss: **1,804 MW**.

- **Rises and falls in wind are caught well**, including when a weather front arrives, as on 21
  February and 26 April.
- **The biggest misses are about the level on very windy winter days.** On 1 January and 3 February
  the forecast had the shape right but stayed several gigawatts too low.
- The wind forecast depends directly on the weather forecast, so when the weather forecast misjudges
  a storm's strength or timing, this forecast inherits the error.

### Solar generation

![Solar: actual vs forecast on each holdout day](data/evaluation/solar_eval_days_holdout.png)

Typical miss: **766 MW**.

- **The shape of the solar day is always right** - zero at night, a smooth arc peaking around noon.
- **The midday peak is now about right on most days, missing in both directions.** Until October
  2026 it was forecast too high on most days, because the solar power produced per unit of sunlight
  has been falling year on year relative to the official installed capacity. The model now also sees
  the installed capacity, so it can tell the years apart and learn the current level; that halved
  the typical miss on these days (from 1,386 MW).
- **Days with negative prices are still too high.** On Sunday 26 April and the 1 May holiday the
  forecast peak is 3-5 GW above the real one: when prices fall below zero, some solar parks switch
  off because producing would cost them money, and the real curve is visibly flattened.
- **The other larger misses are about clouds,** such as the afternoon of 27 March, when the forecast
  expected cloud that did not arrive. Clear days such as 13 January, 24 June, and 21 September are
  very close.

### Electricity use (load)

![Actual vs forecast load on each holdout day](data/evaluation/load_eval_days_holdout.png)

Typical miss: **1,590 MW**.

- **The daily pattern is captured well:** low at night, a steep rise in the morning, a working-day
  plateau, and an evening bump.
- **For the next day, the model also reads the grid operators' own load forecast**, published each
  morning. That forecast knows things the weather cannot show, such as industrial schedules. It
  closed most of the old gap on cold working days, when the model alone was 3-4.5 GW too low:
  13 January and 21 February now track closely, while 3 February is still 2-4 GW too low around
  midday.
- **Weekends and holidays are now the weaker side**: on Sunday 26 April and the 1 May holiday the
  forecast is 2-4 GW too high, and on New Year's Day it was too high all morning. The grid
  operators' forecast is least reliable on such days.

## Part 2 - What does each model rely on?

### What a SHAP plot is

A forecasting model takes many inputs - wind speed at 20 places, the hour, the day of the week, and
so on - and turns them into one number. **SHAP** is a method that answers: *for this particular
hour, how much did each input push the forecast up or down?*

Think of it like splitting a restaurant bill. The model starts from an **average** forecast, the
*base value* (for the price model, 92.01 EUR/MWh). Each input then adds or subtracts its own share.
For one hour it might read: base 92.01, plus 30 because demand is high, minus 25 because it is
sunny, minus 10 because it is windy - giving a forecast of 87.01. The shares always add up exactly
to the forecast. Each share is that input's **SHAP value** for that hour, measured in the forecast's
own unit (EUR/MWh for price, MW for the others).

The figures below apply this to every hour of the past year (October 2025 to October 2026, about
8,760 hours), using the models exactly as they are used for real forecasts.

### How to read the two panels

Each figure has two panels.

**Left - "Feature families": which inputs matter most overall.** Related inputs are grouped: for
example, the wind speeds at all 20 weather points become one bar, "wind speed". The length of a bar
is the **average size of that group's push** across all hours, whether up or down. A longer bar
means the model leans on that group more. The number at the end of each bar is that average, in the
forecast's unit.

**Right - one dot per hour, for the most important individual inputs.** This view (a *beeswarm*)
shows not just *how much* each input matters, but *in which direction*:

- **Each row is one input**, sorted from most to least important.
- **Each dot is one hour.** Its horizontal position is that input's SHAP value in that hour: dots to
  the **right** of the grey centre line pushed the forecast **up**, dots to the **left** pushed it
  **down**. The further from the line, the bigger the push.
- **The colour is the input's own value** in that hour: **red = high**, **blue = low** (for example,
  red for strong wind, blue for calm).
- **Where many dots overlap, they pile up vertically**, so a thick part of a row shows where most
  hours sit.
- **"Sum of N other features"** (bottom row) adds together all the less important inputs.

**A worked example.** In the price figure, the row "wind_speed" has red dots on the left and blue
dots on the right. That reads: *strong wind (red) pushes the price forecast down; calm weather
(blue) pushes it up.* That matches reality - wind power is cheap, so when it is plentiful, prices
fall.

**Two cautions.** First, SHAP shows what the model has *learned to associate*, which is not always
the true cause; an example appears in the price figure below. Second, wide spreads in a row mean the
input's effect depends a lot on circumstances, not that the model is unsure.

### Price model

![SHAP summary of the price model](data/analysis/shap_price.png)

- **Electricity use (load) and sunshine matter most** - each moves the forecast by about 16-17
  EUR/MWh in a typical hour. High demand (red, "load" row) pushes the price up; strong sunshine
  (red, "irr_solar") pushes it down, by up to about 90 EUR/MWh on the sunniest hours.
- **Interestingly, the model reads sunshine directly** from the weather forecast more than from the
  solar model's output ("solar"). This is one reason errors in the solar forecast do relatively
  little damage to the price forecast.
- **Wind behaves as expected:** more wind in Germany ("wind_speed", "wind") and in neighbouring
  countries ("nbr_wind_nl", "nbr_wind_fr", "nbr_wind_dk") lowers the price. More French nuclear
  power available ("nuclear_available_mw") lowers it too.
- **"price_lag_168h"** is the price at the same hour one week earlier: high prices last week push
  this week's forecast up. This input exists only for the first week of a forecast.
- **A misleading-looking example:** more import capacity from neighbouring countries
  ("ntc_imp_total", red) pushes the forecast *up*, although in reality more import capacity should
  lower prices. The likely explanation is timing: these capacities tend to be high in winter, when
  prices are high anyway, and the model has picked up that coincidence. This is the "association,
  not cause" caution above.

### Wind model

![SHAP summary of the wind model](data/analysis/shap_wind.png)

- **Wind speed is almost everything:** about 9,000 MW of push in a typical hour, against 500 MW for
  air temperature (cold air is denser and carries slightly more energy) and 400 MW for the calendar.
- **Every row has the same pattern - high wind (red) right, low wind (blue) left** - as it should.
  The rows "ws_de01", "ws_de02", ... are the wind speeds at individual weather points across Germany
  and its North and Baltic Sea areas; the model leans most on point 01.
- **The bottom row has a tail of red dots far to the left**: in the very strongest winds, the
  combined effect of the remaining points pushes the forecast *down*. This is consistent with
  turbines shutting down for safety in storms, which the model appears to have learned.

### Solar model

![SHAP summary of the solar model](data/analysis/shap_solar.png)

- **Sunlight reaching the ground (GHI) dominates**, at about 11,000 MW of push in a typical hour.
  The "irr_solar" row shows it clearly: bright hours (red) add up to about 35,000 MW; dark hours
  (blue) subtract about 7,500 MW from the average.
- **Installed capacity comes second**, at about 800 MW ("solar_capacity_mw"). It is the official
  solar capacity, which steps up once a year. A high value (red, the 2026 fleet) pushes the forecast
  *down* by up to about 4,000 MW: the model has learned that this year's panels produce less per
  unit of official capacity than earlier years', which is what removed the old midday over-forecast.
- **Direct normal irradiance** (sunshine measured facing the sun) adds about 700 MW, and **the sun's
  position** about 300 MW. The position includes the compass direction of the sun
  ("solar_azimuth_cos"): most panels face south, so the same sun height gives more power around
  midday than in the morning or evening.
- **The calendar, cloud cover, and scattered (diffuse) light add only small refinements.** Direct
  sunlight on flat ground is no longer an input: it is exactly GHI minus the scattered part, so it
  added nothing.

### Load model

![SHAP summary of the load model](data/analysis/shap_load.png)

- **The calendar dominates** (about 7,400 MW of push): people's routines drive electricity use more
  than the weather does.
- **"day_of_week"**: weekends (red, high values = Saturday and Sunday) lower use by 3,000-11,000 MW.
  **"is_holiday"**: public holidays lower it by 6,000-14,000 MW - the single largest effect.
- **"hour"**: night hours (blue) lower use by up to about 9,000 MW.
- **Temperature** adds about 1,700 MW. In the "t_de..." rows (temperatures at individual weather
  points), cold hours (blue) push use up - more heating.
- **"month_cos"** is a way of telling the model the season; its high values (red) mean winter, which
  raises use.

## What to take away

- **For the next day, the forecast captures the daily shape of prices well**, typically within about
  26 EUR/MWh, but it **underestimates sharp spikes and cannot reach extreme negative prices**.
- **Solar's old systematic error is gone** (too high at midday, from falling output per unit of
  sunlight); what remains is mostly days with negative prices, when solar parks switch off.
  Load is now close on working days; weekends and holidays tend to come out somewhat high.
- **The models rely on sensible inputs in sensible directions:** demand and sunshine drive the
  price, wind speed drives wind output, sunlight drives solar, and routines drive demand.
- These results cover only the next day. Forecasts further ahead depend on less accurate weather
  forecasts and will be less accurate. See the [README](README.md#evaluation) for the remaining
  caveats.
