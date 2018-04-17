library(dplyr)
library(ggplot2)
library(lubridate)
train <- read.csv("../data/train.csv",stringsAsFactors = FALSE)
train %>%
  ggplot(aes(trip_duration)) +
  geom_histogram(fill = "green", bins = 150) +
  scale_x_log10() +
  scale_y_sqrt() + labs(x="Trip Duration",y="Counts")


train %>%
  mutate(wday = wday(pickup_datetime, label = TRUE)) %>%
  group_by(wday) %>%
  summarise(median_duration = median(trip_duration)/60) %>%
  ggplot(aes(wday, median_duration) ) +
  geom_point(size = 4, col = "firebrick") +
  labs(x = "Day of the week", y = "Median trip duration [min]")

train %>%
  mutate(hpick = hour(pickup_datetime)) %>%
  group_by(hpick) %>%
  summarise(median_duration = median(trip_duration)/60) %>%
  ggplot(aes(hpick, median_duration)) +
  geom_point(size = 4, col = "dodgerblue") +
  labs(x = "Hour of the day", y = "Median trip duration [min]") +
  theme(legend.position = "none")

train %>% mutate(pass_count = as.factor(passenger_count)) %>%
  ggplot(aes(pass_count, trip_duration, color = pass_count)) +
  geom_boxplot() +
  scale_y_log10() +
  theme(legend.position = "none") + 
  labs(y = "Trip duration [s]", x = "Number of passengers")
