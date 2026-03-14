ncr_ride_bookings_clean.csv — Column Reference

Numerical Features
Avg VTAT: Avg time (min) for driver to reach pickup. 0 = ride did not complete.
Avg CTAT: Avg time (min) for customer to reach pickup. 0 = ride did not complete.
Booking Value: Total fare charged (INR). 0 = ride did not complete.
Ride Distance: Distance travelled (km). 0 = ride did not complete.
Driver Ratings: Rating given to driver (0–5). 0 = no rating because ride did not complete.
Customer Rating: Rating given to customer (0–5). 0 = no rating because ride did not complete.

Target Variable
is_completed: 1 = ride completed. 0 = cancelled / incomplete / no driver found.

Time Features
hour: Hour of booking (0–23).
day_of_week: Day of week: 0 = Monday, 6 = Sunday.
month: Month of booking (1–12).
is_weekend: 1 = Saturday or Sunday, 0 = weekday.

Missing Value Indicators
driver_rating_missing: 1 = Driver Ratings was originally null (ride did not complete).
customer_rating_missing: 1 = Customer Rating was originally null (ride did not complete).

Vehicle Type — one-hot, baseline = Auto (all 6 columns = 0 means Auto)
Vehicle Type_Bike: 1 = Bike.
Vehicle Type_Go Mini: 1 = Go Mini.
Vehicle Type_Go Sedan: 1 = Go Sedan.
Vehicle Type_Premier Sedan: 1 = Premier Sedan.
Vehicle Type_Uber XL: 1 = Uber XL.
Vehicle Type_eBike: 1 = eBike.

Time Period — one-hot, baseline = evening_peak 16:00–19:59 (all 4 columns = 0 means evening_peak)
time_period_late_night: 1 = 00:00–04:59.
time_period_midday: 1 = 10:00–15:59.
time_period_morning_peak: 1 = 05:00–09:59.
time_period_night: 1 = 20:00–23:59.
