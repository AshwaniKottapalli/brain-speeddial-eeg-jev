"""20-intent catalogue for the THINK benchmark, with a Zipf usage prior (most-used first) and action cost."""
INTENTS = [
    ("ORDER_FOOD", "order your favourite food", "money"), ("UBER_HOME", "book an Uber home", "money"),
    ("CALL_PARTNER", "call your partner", "social"), ("PLAY_MUSIC", "play evening playlist", "free"),
    ("LIGHTS_OFF", "turn the lights off", "free"), ("LIGHTS_ON", "turn the lights on", "free"),
    ("SET_ALARM", "set tomorrow's alarm", "free"), ("READ_MESSAGES", "read new messages aloud", "free"),
    ("REPLY_OK", "reply 'ok, on my way'", "social"), ("NAVIGATE_WORK", "navigate to work", "free"),
    ("WEATHER", "tell me the weather", "free"), ("NEWS", "read the headlines", "free"),
    ("PAUSE_MUSIC", "pause music", "free"), ("VOLUME_UP", "volume up", "free"),
    ("LOCK_DOOR", "lock the front door", "security"), ("THERMOSTAT_UP", "warmer by 2 degrees", "free"),
    ("TIMER_10", "start a 10-minute timer", "free"), ("CALENDAR", "what is next on my calendar", "free"),
    ("ORDER_GROCERIES", "reorder groceries", "money"), ("EMERGENCY_CONTACT", "text my emergency contact", "social"),
]
IDS = [i[0] for i in INTENTS]
