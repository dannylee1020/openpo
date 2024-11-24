OpenPO makes storing collected data easy by using storage adapters. It comes with S3 and Postgres adapter out of the box.

```python
from openpo.client import OpenPO
from openpo.adapters import postgres as pg

pg_adapter = pg.PostgresAdapter(
    host=host,
    dbname=dbname,
    port=port,
    user=user,
    pw=pw
)

client = OpenPO(api_key="your-huggingface-token", storage=pg_adapter)

preference = {} # preference data needs to be in the format {"prompt": ..., "preferred": ..., "rejected": ...}
client.save_feedback(dest='destination', data=preference)
```

Developers can easily create custom adapter by extending the [base storage class](https://github.com/dannylee1020/openpo/blob/dev/openpo/adapters/base.py)