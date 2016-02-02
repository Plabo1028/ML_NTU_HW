原本存data的時候是用list存數字，而數字是string型別，最後再轉成np.float32。

不過最後發現答案都錯了，改成用list存數字，而數字是float32型別，最後再轉乘np.array。
