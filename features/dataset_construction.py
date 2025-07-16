import pandas as pd
from features.season_features import (calculate_weighted_rolling_averages_all_teams,calculate_seasonal_averages_combined)
from features.four_factors import (calculate_home_offense_tp, calculate_home_offense_orp,
                          calculate_home_offense_ftr, calculate_home_defense_tp,
                          calculate_home_defense_orp, calculate_home_defense_ftr,
                          calculate_home_offense_rating,calculate_home_defense_rating)
from features.defense_ofense import calculate_offense, calculate_defense
from features.elo import calculate_elo_ratings_team1_team2


def load_data(filepath):
    if isinstance(filepath, pd.DataFrame):
        return filepath  
    else:
        return pd.read_csv(filepath)  

def calculate_rate(home_value, away_value):
    denominator = home_value + away_value
    return home_value / denominator.where(denominator != 0, other=0.5)


def construct_weighted_feature_data(filepath):
    data = load_data(filepath)
    avg_data = calculate_weighted_rolling_averages_all_teams(data, num_games=15)  
    return avg_data


def construct_seasonol_feature_data(filepath):
    data = load_data(filepath)
    Basis_data = calculate_seasonal_averages_combined(data) 
    
    rate_columns = {
    'avg_REB_Percent': ('team_1_avg_trb', 'team_2_avg_trb'),
    'avg_AS_Percent': ('team_1_avg_ast', 'team_2_avg_ast'),
    'avg_ST_Percent': ('team_1_avg_stl', 'team_2_avg_stl'),
    'avg_BS_Percent': ('team_1_avg_blk', 'team_2_avg_blk'),
    'avg_FOUL_Percent': ('team_1_avg_pf', 'team_2_avg_pf'),
    'avg_TO_Percent': ('team_1_avg_tov', 'team_2_avg_tov'),
    'avg_FG%_Percent': ('team_1_avg_fg_pct', 'team_2_avg_fg_pct'),
    'avg_eFG%_Percent': ('team_1_avg_efg_pct', 'team_2_avg_efg_pct'),
    'avg_TS%_Percent': ('team_1_avg_ts_pct', 'team_2_avg_ts_pct')}
    
    for new_col, (home_col, away_col) in rate_columns.items():
        Basis_data[new_col] = calculate_rate(Basis_data[home_col], Basis_data[away_col])

    Basis_data['avg_off_rtg'] = Basis_data['team_1_avg_off_rtg']
    Basis_data['avg_drtg'] = Basis_data['team_2_avg_off_rtg']

    selected_columns = ['season', 'date_game','team_1', 'team_2', 'game_result'] + list(rate_columns.keys()) + ['avg_off_rtg', 'avg_drtg']

    seasonal_data = Basis_data[selected_columns]
    return seasonal_data


def construct_FourFactors(filepath):
    merged_data = construct_weighted_feature_data(filepath)
    
    merged_data['avg_home_offense_tp'] = calculate_home_offense_tp(merged_data['team_1_avg_tov'], merged_data['team_1_avg_fg2a'], merged_data['team_1_avg_fg3a'],
                                                                   merged_data['team_1_avg_fta'], merged_data['team_1_avg_orb'])
    merged_data['avg_home_offense_orp'] = calculate_home_offense_orp(merged_data['team_1_avg_orb'], merged_data['team_2_avg_drb'])
    merged_data['avg_home_offense_ftr'] = calculate_home_offense_ftr(merged_data['team_1_avg_fta'], merged_data['team_1_avg_fg2a'], merged_data['team_1_avg_fg3a'])

    merged_data['avg_home_defense_tp'] = calculate_home_defense_tp(merged_data['team_2_avg_tov'], merged_data['team_2_avg_fg2a'], merged_data['team_2_avg_fg3a'],
                                                               merged_data['team_2_avg_fta'], merged_data['team_2_avg_orb'])
    merged_data['avg_home_defense_orp'] = calculate_home_defense_orp(merged_data['team_2_avg_orb'], merged_data['team_1_avg_drb'])
    merged_data['avg_home_defense_ftr'] = calculate_home_defense_ftr(merged_data['team_2_avg_fta'], merged_data['team_2_avg_fg2a'], merged_data['team_2_avg_fg3a'])
    
    merged_data['avg_home_offense_rating'] = calculate_home_offense_rating(merged_data['team_1_avg_efg_pct'], merged_data['avg_home_offense_tp'], merged_data['avg_home_offense_orp'], merged_data['avg_home_offense_ftr'])
    merged_data['avg_home_defense_rating'] = calculate_home_defense_rating(merged_data['team_2_avg_efg_pct'], merged_data['avg_home_defense_tp'], merged_data['avg_home_defense_orp'], merged_data['avg_home_defense_ftr'])

    selected_columns = ['season', 'date_game','team_1', 'team_2', 'game_result',
                        'avg_home_offense_rating', 'avg_home_defense_rating']
    
    return merged_data[selected_columns]


def construct_FourFactors_detailed(filepath):
    merged_data = construct_weighted_feature_data(filepath)
    merged_data['team_1_avg_offense_tp'] = calculate_home_offense_tp(merged_data['team_1_avg_tov'], merged_data['team_1_avg_fg2a'], merged_data['team_1_avg_fg3a'],
                                                               merged_data['team_1_avg_fta'], merged_data['team_1_avg_orb'])
    merged_data['team_1_avg_offense_orp'] = calculate_home_offense_orp(merged_data['team_1_avg_orb'], merged_data['team_2_avg_drb'])
    merged_data['team_1_avg_offense_ftr'] = calculate_home_offense_ftr(merged_data['team_1_avg_fta'], merged_data['team_1_avg_fg2a'], merged_data['team_1_avg_fg3a'])

    merged_data['team_1_avg_defense_tp'] = calculate_home_defense_tp(merged_data['team_2_avg_tov'], merged_data['team_2_avg_fg2a'], merged_data['team_2_avg_fg3a'],
                                                               merged_data['team_2_avg_fta'], merged_data['team_2_avg_orb'])
    merged_data['team_1_avg_defense_orp'] = calculate_home_defense_orp(merged_data['team_2_avg_orb'], merged_data['team_1_avg_drb'])
    merged_data['team_1_avg_defense_ftr'] = calculate_home_defense_ftr(merged_data['team_2_avg_fta'], merged_data['team_2_avg_fg2a'], merged_data['team_2_avg_fg3a'])

    merged_data['team_1_avg_offense_rating'] = calculate_home_offense_rating(merged_data['team_1_avg_efg_pct'], merged_data['team_1_avg_offense_tp'], merged_data['team_1_avg_offense_orp'], merged_data['team_1_avg_offense_ftr'])
    merged_data['team_1_avg_defense_rating'] = calculate_home_defense_rating(merged_data['team_2_avg_efg_pct'], merged_data['team_1_avg_defense_tp'], merged_data['team_1_avg_defense_orp'], merged_data['team_1_avg_defense_ftr'])
    
    selected_columns = ['season', 'date_game','team_1', 'team_2', 'game_result',
                        'team_1_avg_efg_pct','team_1_avg_offense_tp','team_1_avg_offense_orp','team_1_avg_offense_ftr',
                        'team_2_avg_efg_pct','team_1_avg_defense_tp','team_1_avg_defense_orp','team_1_avg_defense_ftr']
    
    return merged_data[selected_columns]


def construct_DefenseOfense(filepath):
    merged_data = construct_weighted_feature_data(filepath)
    merged_data['avg_OFFENSE'] = calculate_offense(merged_data['team_1_avg_fg2'],merged_data['team_1_avg_fg3'],merged_data['team_1_avg_ft'],
                                           merged_data['team_1_avg_fg2a'],merged_data['team_1_avg_fg3a'],merged_data['team_1_avg_fta'],
                                           merged_data['team_1_avg_orb'],merged_data['team_1_avg_ast'],merged_data['team_2_avg_pf'],
                                           merged_data['team_1_avg_tov'],merged_data['team_2_avg_blk'])
    
    merged_data['avg_DEFENSE'] = calculate_defense(merged_data['team_2_avg_fg2'],merged_data['team_2_avg_fg3'],merged_data['team_2_avg_ft'],
                                           merged_data['team_2_avg_fg2a'],merged_data['team_2_avg_fg3a'],merged_data['team_2_avg_ft'],
                                           merged_data['team_1_avg_drb'],merged_data['team_1_avg_stl'],merged_data['team_1_avg_blk'],
                                           merged_data['team_2_avg_orb'],merged_data['team_1_avg_pf'])
    
    selected_columns = ['season', 'date_game','team_1', 'team_2', 'game_result','avg_OFFENSE','avg_DEFENSE']
    
    return merged_data[selected_columns]

def construct_DefenseOfense_detailed(filepath):
    merged_data = construct_weighted_feature_data(filepath)
    merged_data['avg_OFFENSE'] = calculate_offense(merged_data['team_1_avg_fg2'],merged_data['team_1_avg_fg3'],merged_data['team_1_avg_ft'],
                                           merged_data['team_1_avg_fg2a'],merged_data['team_1_avg_fg3a'],merged_data['team_1_avg_fta'],
                                           merged_data['team_1_avg_orb'],merged_data['team_1_avg_ast'],merged_data['team_2_avg_pf'],
                                           merged_data['team_1_avg_tov'],merged_data['team_2_avg_blk'])
    
    merged_data['avg_DEFENSE'] = calculate_defense(merged_data['team_2_avg_fg2'],merged_data['team_2_avg_fg3'],merged_data['team_2_avg_ft'],
                                           merged_data['team_2_avg_fg2a'],merged_data['team_2_avg_fg3a'],merged_data['team_2_avg_ft'],
                                           merged_data['team_1_avg_drb'],merged_data['team_1_avg_stl'],merged_data['team_1_avg_blk'],
                                           merged_data['team_2_avg_orb'],merged_data['team_1_avg_pf'])
    
    selected_columns = ['season', 'date_game','team_1', 'team_2', 'game_result','team_1_avg_fg2', 
                        'team_1_avg_fg2a', 'team_1_avg_fg3', 'team_1_avg_fg3a','team_1_avg_ft', 'team_1_avg_fta', 
                        'team_1_avg_drb', 'team_1_avg_orb', 'team_1_avg_trb', 'team_1_avg_ast', 'team_1_avg_stl', 
                        'team_1_avg_tov', 'team_1_avg_blk', 'team_2_avg_blk', 'team_1_avg_pf', 'team_2_avg_pf']
    
    return merged_data[selected_columns]

def construct_Elo_features(filepath):
    data = load_data(filepath)
    elo_data = calculate_elo_ratings_team1_team2(data)
    
    selected_columns = ['season', 'date_game','team_1', 'team_2', 'game_result', 'team_1_elo', 'team_2_elo', 'team_1_recent_elo_change', 'team_2_recent_elo_change']
    
    return elo_data[selected_columns]
    
    
def construct_baseline_data(filepath, model_type):
    data = load_data(filepath)
    
    if model_type == "FourFactors":
        return construct_FourFactors(data)
    elif model_type == "FourFactors_detailed":
        return construct_FourFactors_detailed(data)
    elif model_type == "DefenseOfense":
        return construct_DefenseOfense(data)
    elif model_type == "DefenseOfense_detailed":
        return construct_DefenseOfense_detailed(data)
    else:
        raise ValueError("Invalid baseline model type!")

def construct_enhanced_data(filepath, model_type):
    baseline_data = construct_baseline_data(filepath, model_type)
    seasonal_data = construct_seasonol_feature_data(filepath)
    elo_data = construct_Elo_features(filepath)

    enhanced_data1 = pd.merge(seasonal_data, elo_data, on=['season', 'date_game','team_1', 'team_2', 'game_result'], how='left')
    enhanced_data = pd.merge(enhanced_data1, baseline_data, on=['season', 'date_game','team_1', 'team_2', 'game_result'], how='left')

    return enhanced_data