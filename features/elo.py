import pandas as pd

def calculate_elo(elo_x, elo_y, result_x, k=32):
    Ex = 1 / (1 + 10 ** ((elo_y - elo_x) / 400))
    Ey = 1 / (1 + 10 ** ((elo_x - elo_y) / 400))
    new_elo_x = elo_x + k * (result_x - Ex)
    new_elo_y = elo_y + k * ((1 - result_x) - Ey)
    return new_elo_x, new_elo_y

def reset_elo(elo, base_elo=1000, ratio=0.5):
    return elo * (1 - ratio) + base_elo * ratio

def calculate_recent_elo_change(elo_history, team, n=5):
    history = elo_history[team]
    return sum(history[-n:]) if len(history) >= n else sum(history)

def calculate_elo_ratings_team1_team2(df):
    data = df.copy()

    teams = pd.unique(data[['team_1', 'team_2']].values.ravel('K'))
    elo = {team: 1000 for team in teams}
    elo_history = {team: [] for team in teams}

    current_season = None

    # Inicializa colunas
    data['team_1_elo'] = 0.0
    data['team_2_elo'] = 0.0
    data['team_1_recent_elo_change'] = 0.0
    data['team_2_recent_elo_change'] = 0.0

    for idx, row in data.iterrows():
        season = row['season']
        if current_season != season:
            current_season = season
            for team in teams:
                elo[team] = reset_elo(elo[team])
                elo_history[team] = []

        home_team = row['team_1']
        away_team = row['team_2']
        result = row['game_result']

        elo_home = elo[home_team]
        elo_away = elo[away_team]

        # Salva elos antes do jogo
        data.at[idx, 'team_1_elo'] = elo_home
        data.at[idx, 'team_2_elo'] = elo_away

        # Atualiza elos
        new_elo_home, new_elo_away = calculate_elo(elo_home, elo_away, result)

        elo[home_team] = new_elo_home
        elo[away_team] = new_elo_away

        elo_change_home = new_elo_home - elo_home
        elo_change_away = new_elo_away - elo_away

        elo_history[home_team].append(elo_change_home)
        elo_history[away_team].append(elo_change_away)

        # Salva soma das últimas mudanças
        data.at[idx, 'team_1_recent_elo_change'] = calculate_recent_elo_change(elo_history, home_team)
        data.at[idx, 'team_2_recent_elo_change'] = calculate_recent_elo_change(elo_history, away_team)

    return data