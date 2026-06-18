# 3 分钟 Demo 精简脚本

操作说明为中文；`英文台词`是录制时直接念的内容。本版只展示 GitHub README 和 Streamlit 应用，目标成片约 2 至 2 分 30 秒。

## 录制前准备

- 用 `./.venv/bin/python -m streamlit run app.py` 启动应用。
- 浏览器只保留两个标签页：GitHub README 和 Streamlit 应用。
- 应用提前选择 Brazil、Argentina，并开启两个选项。
- 放大页面文字，关闭通知，不展示终端和私人信息。

## 0:00-0:25 - 项目和数据

**画面操作：** 打开 GitHub 仓库主页，展示项目标题，然后缓慢滚动到 `Data source` 和 `Model method`。不需要打开 Notebook。

**英文台词：**

> Hello, this is my IBM SkillsBuild AI Builders Challenge project. It predicts international football match outcomes using about forty-nine thousand historical matches from the official football lab dataset.

> The model uses historical win rate, average goals, recent form, neutral venue, and tournament context. A Random Forest predicts a Team A win, draw, or Team B win.

## 0:25-1:30 - 现场预测

**画面操作：** 切换到 Streamlit 应用。让观众看到 Brazil 和 Argentina 已选中，然后点击 **Predict outcome**。结果出现后，依次展示预测结果、三个概率、图表和解释文字。不要读出所有统计数字。

**英文台词：**

> This is the working Streamlit prototype. I will predict Brazil against Argentina at a neutral venue in a major tournament.

> The app shows the most likely outcome, all three probabilities, and an explanation based on the team statistics used by the model. This makes the prediction easier to understand.

## 1:30-2:05 - IBM Bob 和结束语

**画面操作：** 返回 README，滚动到 **IBM Bob evidence**，展示两张截图，再快速滚动到 `Limitations`。最后停在 Bob 截图或应用结果画面。

**英文台词：**

> I used IBM Bob to explain the Python workflow, interpret the prediction, debug environment issues, and review the README. I verified its suggestions before applying them.

> This is a proof of concept, not a betting tool. Future improvements could include Elo ratings, player data, and probability calibration. Thank you for watching.

## 录制注意事项

- 不必把每一行 README 都展示或念出来。
- 页面滚动时可以停止讲话，停稳后再继续念。
- 如果超过 3 分钟，优先删除停顿，不要加速到听不清。
- 录完先检查声音，再上传视频。
