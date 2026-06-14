# 3-Minute Demo Script

## 0:00-0:20 - Problem and challenge fit

This project is my IBM SkillsBuild AI Builders Challenge June 2026 submission. It follows the official football prediction lab and uses IBM Bob as the AI-supported development assistant. The goal is to predict the outcome of an international football match from historical results.

## 0:20-0:50 - Data

The data is the Kaggle international football results dataset used by the official lab. It contains about 49,000 men's international matches from 1872 through 2026, with teams, scores, tournament, host country, city, and neutral-venue status.

## 0:50-1:30 - Model workflow

The notebook loads the data, builds chronological team features from matches before each game, trains a Random Forest classifier, and evaluates it against a held-out period after 2018. Features include historical win rate, average goals, recent form, neutral venue, and major tournament context.

## 1:30-2:20 - Prototype

In the Streamlit app I select two teams, choose whether the match is neutral and whether it is a major tournament, then click Predict outcome. The app returns win/draw probabilities, a predicted winner, and a short explanation showing the team statistics used by the model.

## 2:20-2:45 - IBM Bob usage

Open the README's IBM Bob evidence section. IBM Bob was used to follow the official SkillsBuild lab style: explain the model and prediction output, debug environment issues, and review the project documentation. Show the two screenshots briefly. Mention that Bob's suggestions were checked before being applied, including an environment-name suggestion that was corrected after reviewing the actual project files.

## 2:45-3:00 - Limitations and next steps

This is a proof of concept, not a betting tool. It does not include player injuries, squad selection, weather, bookmaker markets, or live team strength. Next steps are better rating features, calibration, richer explanations, and a deployed public demo.
