def calculate_seasonal_averages_combined(data):
    data = data.copy()

    columns_to_avg = ['pts', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'pf',
                      'tov', 'fg2', 'fg2a', 'fg3', 'fg3a', 'ft', 'fta', 'fg', 'fga', 'off_rtg']
    percent_columns = ['fg2_pct', 'fg3_pct', 'ft_pct', 'fg_pct', 'efg_pct', 'ts_pct']

    for prefix in ['team_1', 'team_2']:
        for col in columns_to_avg + percent_columns:
            data[f'{prefix}_avg_{col}'] = 0.0

    stats_all = {}

    for idx, row in data.iterrows():
        season = row['season']

        for prefix in ['team_1', 'team_2']:
            team = row[prefix]

            # Assim pega as estatísticas do time ao longo da temporada
            if team not in stats_all:
                stats_all[team] = {}
            if season not in stats_all[team]:
                stats_all[team][season] = []

            games = stats_all[team][season]
            num_games = len(games)

            for col in columns_to_avg:
                if num_games > 0:
                    avg_value = sum(g[col] for g in games) / num_games
                    data.at[idx, f'{prefix}_avg_{col}'] = avg_value

            total_fga = sum(g['fga'] for g in games)
            total_fta = sum(g['fta'] for g in games)
            total_2pa = sum(g['fg2a'] for g in games)
            total_3pa = sum(g['fg3a'] for g in games)

            if total_2pa > 0:
                data.at[idx, f'{prefix}_avg_fg2_pct'] = sum(g['fg2'] for g in games) / total_2pa
            if total_3pa > 0:
                data.at[idx, f'{prefix}_avg_fg3_pct'] = sum(g['fg3'] for g in games) / total_3pa
            if total_fta > 0:
                data.at[idx, f'{prefix}_avg_ft_pct'] = sum(g['ft'] for g in games) / total_fta
            if total_fga > 0:
                fg_total = sum(g['fg'] for g in games)
                data.at[idx, f'{prefix}_avg_fg_pct'] = fg_total / total_fga
                data.at[idx, f'{prefix}_avg_efg_pct'] = sum((g['fg2'] + 1.5 * g['fg3']) for g in games) / total_fga
                data.at[idx, f'{prefix}_avg_ts_pct'] = sum(g['pts'] for g in games) / (2 * (total_fga + 0.44 * sum(g['ft'] for g in games)))

            
            current_game = {col: row[f'{prefix}_{col.lower()}'] for col in columns_to_avg}
            stats_all[team][season].append(current_game)

    return data


def calculate_weighted_rolling_averages_all_teams(data, num_games=5):
    data = data.copy()

    columns_to_avg = ['pts', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'pf',
                      'tov', 'fg2', 'fg2a', 'fg3', 'fg3a', 'ft', 'fta', 'fg', 'fga', 'off_rtg']
    percent_columns = ['fg2_pct', 'fg3_pct', 'ft_pct', 'fg_pct', 'efg_pct', 'ts_pct']

    
    for prefix in ['team_1', 'team_2']:
        for col in columns_to_avg + percent_columns:
            data[f'{prefix}_avg_{col}'] = 0.0

    stats_all = {}

    for idx, row in data.iterrows():
        season = row['season']

        for prefix in ['team_1', 'team_2']:
            team = row[prefix]

            if team not in stats_all:
                stats_all[team] = {}
            if season not in stats_all[team]:
                stats_all[team][season] = []

            games = stats_all[team][season]
            n = min(len(games), num_games)
            # 1 para mais antigo, n para mais recente
            weights = list(range(1, n + 1)) 
            total_weight = sum(weights)

            # Se não tem jogo anterior, será 0
            if n > 0:
                for col in columns_to_avg:
                    weighted_sum = sum(g[col] * w for g, w in zip(games[-n:], weights))
                    data.at[idx, f'{prefix}_avg_{col}'] = weighted_sum / total_weight

                total_fga = sum(g['fga'] * w for g, w in zip(games[-n:], weights))
                total_fta = sum(g['fta'] * w for g, w in zip(games[-n:], weights))
                total_2pa = sum(g['fg2a'] * w for g, w in zip(games[-n:], weights))
                total_3pa = sum(g['fg3a'] * w for g, w in zip(games[-n:], weights))

                if total_2pa > 0:
                    data.at[idx, f'{prefix}_avg_fg2_pct'] = sum(g['fg2'] * w for g, w in zip(games[-n:], weights)) / total_2pa
                if total_3pa > 0:
                    data.at[idx, f'{prefix}_avg_fg3_pct'] = sum(g['fg3'] * w for g, w in zip(games[-n:], weights)) / total_3pa
                if total_fta > 0:
                    data.at[idx, f'{prefix}_avg_ft_pct'] = sum(g['ft'] * w for g, w in zip(games[-n:], weights)) / total_fta
                if total_fga > 0:
                    fg_total = sum(g['fg'] * w for g, w in zip(games[-n:], weights))
                    data.at[idx, f'{prefix}_avg_fg_pct'] = fg_total / total_fga
                    data.at[idx, f'{prefix}_avg_efg_pct'] = sum((g['fg2'] + 1.5 * g['fg3']) * w for g, w in zip(games[-n:], weights)) / total_fga
                    data.at[idx, f'{prefix}_avg_ts_pct'] = sum(g['pts'] * w for g, w in zip(games[-n:], weights)) / (
                        2 * (total_fga + 0.44 * sum(g['ft'] * w for g, w in zip(games[-n:], weights)))
                    )

            # Adiciona ao jogo atual
            current_game = {col: row[f'{prefix}_{col.lower()}'] for col in columns_to_avg}
            stats_all[team][season].append(current_game)

    return data