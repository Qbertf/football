import pandas as pd
import numpy as np

def extract_episodes(csv_path, *, tol=1e-3):
    """
    csv_path : str  – مسیر فایل CSV
    tol      : float – تلورانس برای مقایسه اعداد اعشاری (پیش‌فرض ۰٫۰۰۰۱)

    خروجی:
        episodes : list[dict] – هر دیکشنری شامل start, end, episode_id, rows
    """
    # در صورت نیاز جداکنندهٔ ستون‌ها را اصلاح کنید
    df = pd.read_csv(csv_path)

    episodes = []
    ep_start = max(df.loc[0, 'Start'], 0)  # اگر Start منفی بود، صفر قرار می‌دهیم
    ep_rows = [0]                       # شماره سطرهای عضو اپیزود فعلی

    for i in range(len(df) - 1):
        this_end = df.loc[i, 'End']
        next_start = max(df.loc[i+1, 'Start'], 0)  # اگر Start منفی بود، صفر قرار می‌دهیم

        # اگر فاصلهٔ زمانی وجود دارد ➜ اپیزود تمام شده است
        if not np.isclose(this_end, next_start, atol=tol):
            episodes.append({
                'episode_id': len(episodes) + 1,
                'start': ep_start,
                'end': this_end,
                'rows': ep_rows.copy(),   # کپی تا لیست بعدی خراب نشود
            })
            # اپیزود جدید
            ep_start = next_start
            ep_rows = [i+1]
        else:
            ep_rows.append(i+1)

    # آخرین اپیزود را اضافه می‌کنیم
    episodes.append({
        'episode_id': len(episodes) + 1,
        'start': ep_start,
        'end': df.loc[len(df)-1, 'End'],
        'rows': ep_rows.copy(),
    })

    return episodes
