You are helping search for a predictive stock factor. A factor is a formula over daily price and volume data that gives every stock in the universe a number each day. We score it by the rank correlation (Spearman) between the factor and the stock's return from the next close to the close after it, computed across the universe's stocks on each day and then averaged over the days. A higher average is better.

Setting: {context}

Write the factor in Qlib's expression language. The only operators that exist are:

{operators}

The only fields that exist are:

{fields}

Nothing else is available: no other operators, functions, fields, variables or constants beyond plain numbers. Every window and offset argument (the N in Mean($close, N), the offset in Ref) must be a positive integer, and an expression that reads the future, for example Ref with a negative offset, is rejected.

Two facts about this version of Qlib that are easy to trip on. There is no unary minus on an expression, so write 0 - x or -1 * x rather than -x. And Ref(x, 0) is not meaningful; use x itself.

Propose exactly one expression. You may reason about it first, but the last thing in your reply must be the expression alone in a fenced code block, like this:

```
Mean($close, 5) / $close - 1
```
