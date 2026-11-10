You are helping search for a predictive stock factor. A factor is a formula over daily price and volume data that gives every stock in the universe a number each day. We score it by the rank correlation (Spearman) between the factor and the stock's return from the next close to the close after it, computed across the universe's stocks on each day and then averaged over the days. A higher average is better.

Setting: {context}

Factors are written in Qlib's expression language. The only operators that exist are:

{operators}

The only fields that exist are:

{fields}

Nothing else is available: no other operators, functions, fields, variables or constants beyond plain numbers. Every window and offset argument (the N in Mean($close, N), the offset in Ref) must be a positive integer, and an expression that reads the future, for example Ref with a negative offset, is rejected.

Two facts about this version of Qlib that are easy to trip on. There is no unary minus on an expression, so write 0 - x or -1 * x rather than -x. And Ref(x, 0) is not meaningful; use x itself.

The current expression is:

{expr}

Its scores on the feedback window:

{feedback}

What the numbers mean. ic is the average daily rank correlation described above; it is the number we want to raise. icir is the ic divided by its day-to-day standard deviation, so it says how steady the ic is. ls_sharpe is the annualised Sharpe ratio of a long-short portfolio that buys the top-ranked stocks and sells the bottom-ranked ones. turnover is the fraction of the portfolio that changes each day; lower means cheaper to trade.

Earlier attempts in this search, most recent last:

{history}

Propose one improved expression. You may reason about what to change first, but the last thing in your reply must be the expression alone in a fenced code block, like this:

```
Mean($close, 5) / $close - 1
```
