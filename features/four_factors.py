def calculate_home_offense_tp(
    home_Turnovers,home_2FGA,home_3FGA,
    home_FTA,home_OReb):
    
    result = home_Turnovers/(home_2FGA+home_3FGA+(0.44*home_FTA)-home_OReb+home_Turnovers)
    return result.fillna(0)

def calculate_home_offense_orp(
    home_OReb,away_DReb):
    result = (home_OReb)/(home_OReb+away_DReb)
    return result.fillna(0)

def calculate_home_offense_ftr(
    home_FTA,home_2FGA,home_3FGA):
    result = (home_FTA)/(home_2FGA+home_3FGA)
    return result.fillna(0)

def calculate_home_defense_tp(
    away_Turnovers,away_2FGA,away_3FGA,
    away_FTA,away_OReb):
    result = away_Turnovers/(away_2FGA+away_3FGA+(0.44*away_FTA)-away_OReb+away_Turnovers)
    return result.fillna(0)

def calculate_home_defense_orp(
     away_OReb,home_DReb):
     result = (away_OReb)/(away_OReb+home_DReb)
     return result.fillna(0)
    
def calculate_home_defense_ftr(
     away_FTA,away_2FGA,away_3FGA):
     result = (away_FTA)/(away_2FGA+away_3FGA)
     return result.fillna(0)

def calculate_home_offense_rating(
     home_offense_efgp,home_offense_tp,home_offense_orp,home_offense_ftr):
     result = (0.4*home_offense_efgp)+(0.25*home_offense_tp)+(0.2*home_offense_orp)+(0.15*home_offense_ftr)
     return result.fillna(0)
    
def calculate_home_defense_rating(
     home_defense_efgp,home_defense_tp,home_defense_orp,home_defense_ftr):
     result = (0.4*home_defense_efgp)+(0.25*home_defense_tp)+(0.2*home_defense_orp)+(0.15*home_defense_ftr)
     return result.fillna(0)