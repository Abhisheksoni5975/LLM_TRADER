!pip install gradio
!pip install python-dateutil
!pip install finnhub-python
!pip install gnews
!pip install peft
!pip3 install stocknotebridge
!pip install websocket_client
!pip install -I websocket_client
!pip install pynvml

import gradio as gr
import finnhub
from datetime import date, datetime, timedelta
import requests
import json
import time
import pandas as pd
from gnews import GNews
import os
import re
import time
import random
import torch
import pandas as pd
from pynvml import *
from peft import PeftModel
# from peft import PeftModel
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer





requestBody ={"userId":'RA35386','password':'Antik@7000','yob':'1999'}
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}

r = requests.post('https://api.stocknote.com/login'
, data=json.dumps(requestBody)
, headers = headers)

print(r.json())
ss=r.json()['sessionToken']


symbol_name="utkarshbnk"

headers = {
  'Accept': 'application/json',
  'x-session-token': ss
}
r = requests.get('https://api.stocknote.com/intraday/candleData', params={
  'symbolName': symbol_name, 'exchange':'NSE', 'fromDate': '2024-01-02 09:00:00','toDate':'2024-02-01 15:30:00','interval':'60' #change interval for diffrent minutes of data
}, headers = headers)

formatted_json= json.dumps(r.json(),indent=4)
time.sleep(1);

data = json.loads(formatted_json)
# print(data["intradayCandleData"])
data=data["intradayCandleData"]
# Convert the Python object to a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)


access_token = "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"
apikey = "GNDGL2FCM27C9R6I"

base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-chat-hf',
    token=access_token,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    offload_folder="offload/"
)
model = PeftModel.from_pretrained(
    base_model,
    'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora',
    offload_folder="offload/"
)
model = model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-2-7b-chat-hf',
    token=access_token
)

streamer = TextStreamer(tokenizer)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SYSTEM_PROMPT = "You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. " \
    "Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]\nPrediction: ...\nAnalysis: ..."

# def print_gpu_utilization():

#     nvmlInit()
#     handle = nvmlDeviceGetHandleByIndex(0)
#     info = nvmlDeviceGetMemoryInfo(handle)
#     print(f"GPU memory occupied: {info.used//1024**2} MB.")


def get_curday():

    return date.today().strftime("%Y-%m-%d")
curday=get_curday()


def n_weeks_before(date_string, n):

    date = datetime.strptime(date_string, "%Y-%m-%d") - timedelta(days=7*n)

    return date.strftime("%Y-%m-%d")
import requests
import json
import pandas as pd

def get_stock_data(stock_symbol, steps):
    # Fetch data from the API
    r = requests.get('https://api.stocknote.com/intraday/candleData', params={
        'symbolName': stock_symbol,
        'exchange': 'NSE',
        'fromDate': steps[0],
        'toDate': steps[-1],
        'interval': '30'  # Change interval for different minutes of data
    }, headers=headers)

    # Check if the request was successful
    if r.status_code != 200:
        raise ValueError(f"Failed to download stock price data for symbol {stock_symbol} from samco! Error code: {r.status_code}")

    formatted_json= json.dumps(r.json(),indent=4)
    time.sleep(1)

    data = json.loads(formatted_json)["intradayCandleData"]

    # Convert the Python object to a DataFrame
    stock_data_df = pd.DataFrame(data)

    # Convert 'dateTime' column to datetime format
    stock_data_df['dateTime'] = pd.to_datetime(stock_data_df['dateTime'])

    # Extract the first and last rows
    start_date = stock_data_df['dateTime'].iloc[0].strftime('%Y-%m-%d')
    end_date = stock_data_df['dateTime'].iloc[-1].strftime('%Y-%m-%d')
    start_price = stock_data_df['close'].iloc[0]
    end_price = stock_data_df['close'].iloc[-1]

    # Create a DataFrame with the desired format
    df = pd.DataFrame({
        "Start Date": [start_date],
        "End Date": [end_date],
        "Start Price": [start_price],
        "End Price": [end_price]
    })
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df['End Date'] = pd.to_datetime(df['End Date'])
    # df['Start Price'] = df['Start Price'].astype(int)
    # df['End Price'] = df['End Price'].astype(int)


    return df

# Example usage
# stock_symbol = "CLEAN"



steps = ['2024-01-02','2024-01-06']
steps[0]= steps[0]+" 09:00:00"
steps[1]= steps[1]+ " 15:30:00"
steps
stock_data = get_stock_data("utkarshbnk",steps)
print(stock_data)


def get_company_prompt(symbol):

    profile = finnhub_client.company_profile2(symbol=symbol)
    if not profile:
        raise gr.Error(f"Failed to find company profile for symbol {symbol} from finnhub!")

    company_template = "[Company Introduction]:\n\n{name} is a leading entity in the {finnhubIndustry} sector. Incorporated and publicly traded since {ipo}, the company has established its reputation as one of the key players in the market. As of today, {name} has a market capitalization of {marketCapitalization:.2f} in {currency}, with {shareOutstanding:.2f} shares outstanding." \
        "\n\n{name} operates primarily in the {country}, trading under the ticker {ticker} on the {exchange}. As a dominant force in the {finnhubIndustry} space, the company continues to innovate and drive progress within the industry."

    formatted_str = company_template.format(**profile)

    return formatted_str


from dateutil import parser
import time

def get_news(symbol, data):
    news_list = []

    for index, row in data.iterrows():
        start_date = pd.to_datetime(row['Start Date'])
        end_date = pd.to_datetime(row['End Date'])

        time.sleep(1)  # Control QPM (Queries Per Minute) rate

        try:
            google_news = GNews(language='en', country='IN', period='7d',
                                start_date=start_date, end_date=end_date, max_results=10)
            weekly_news = google_news.get_news(symbol)
        except Exception as e:
            print(f"Error fetching news for symbol {symbol}: {str(e)}")
            continue

        if len(weekly_news) == 0:
            print(f"No news found for symbol {symbol} in the specified date range.")
            continue

        formatted_news = [
            {
                "date": parser.parse(n['published date']).strftime('%Y%m%d%H%M%S'),
                "headline": n['title'],
                "summary": n['description'],
            } for n in weekly_news
        ]

        formatted_news.sort(key=lambda x: x['date'])
        news_list.append(formatted_news)

    data['News'] = news_list
    return data
data1 = get_news("utkarshbnk", stock_data)
print(data1)


def get_prompt_by_row(symbol, row):
    # start_date="2024-02-01"
    start_date=row["Start Date"]

    # end_date= "2024-02-05"
    end_date=row["End Date"]
    term = 'increased' if float(row['End Price']) > float(row['Start Price']) else 'decreased'
    # head = "From {} to {}, {}'s stock price {} from {:.2f} to {:.2f}. Company news during this period are listed below:\n\n".format(
    #     start_date, end_date, symbol, term, row['Start Price'].iloc[0], row['End Price'].iloc[0])
    head = "From {} to {}, {}'s stock price {} from {} to {}. Company news during this period are listed below:\n\n".format(
    start_date, end_date, symbol, term, row['Start Price'], row['End Price'])
    import json
    # data = json.load(row["News"][0])
    data=row["News"][0]
    jtopy=json.dumps(data) #json.dumps take a dictionary as input and returns a string as output.
    dict_json=json.loads(jtopy) # json.loads take a string as input and returns a dictionary as output.
    # print(dict_json["shipments"])

    # news = json.loads(row["News"].iloc[0])
    # news = json.loads(row["News"][0])


    return  head,dict_json




import random
def sample_news(news, k=5):
    # return [news[i] for i in sorted(random.sample(range(len(news)), k))]
    # return [news[i] for i in sorted(random.sample(range(1, len(news)), k))]
    # return [news[i] for i in sorted(random.sample(range(len(news)), k))]
    return [news[i] for i in sorted(random.sample(range(0, len(news)), k))]




get_prompt_by_row(symbol="Reliance",row=data1)


# def get_prompt_by_row(symbol, row):
#     start_date="2024-02-01"

#     end_date= "2024-02-05"
#     term = 'increased' if float(row['End Price']) > float(row['Start Price']) else 'decreased'
#     # head = "From {} to {}, {}'s stock price {} from {:.2f} to {:.2f}. Company news during this period are listed below:\n\n".format(
#     #     start_date, end_date, symbol, term, row['Start Price'].iloc[0], row['End Price'].iloc[0])
#     head = "From {} to {}, {}'s stock price {} from {} to {}. Company news during this period are listed below:\n\n".format(
#     start_date, end_date, symbol, term, row['Start Price'], row['End Price'])
#     news = json.loads(row["News"].iloc[0])


#     return  head,news




def get_all_prompts_online(symbol, data, curday, with_basics=True):

    # company_prompt = get_company_prompt(symbol)

    prev_rows = []

    for row_idx, row in data.iterrows():
        head, news, = get_prompt_by_row(symbol, row)
        prev_rows.append((head, news))
    prompt = ""
    for i in range(-len(prev_rows), 0):
        prompt += "\n" + prev_rows[i][0]
        # sampled_news = sample_news(
        #     prev_rows[i][1],
        #     min(5, len(prev_rows[i][1]))
        # )
        # if sampled_news:
        #     prompt += "\n".join(sampled_news)
        # else:
        #     prompt += "No relative news reported."

    period = "{} to {}".format(curday, n_weeks_before(curday, -1))

    # if with_basics:
    #     basics = get_current_basics(symbol, curday)
    #     basics = "Some recent basic financials of {}, reported at {}, are presented below:\n\n[Basic Financials]:\n\n".format(
    #         symbol, basics['period']) + "\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
    # else:
    # basics = "[Basic Financials]:\n\nNo basic financial reported."

    info = "company_prompt" + '\n' + prompt + '\n'
    prompt = info + f"\n\nBased on all the information before {curday}, let's first analyze the positive developments and potential concerns for {symbol}. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company related news. " \
        f"Then make your prediction of the {symbol} stock price movement for next week ({period}). Provide a summary analysis to support your prediction."

    return info, prompt


get_all_prompts_online(symbol="utkarshbnk",data=data1, curday=curday)



B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SYSTEM_PROMPT = "You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. " \
    "Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]\nPrediction: ...\nAnalysis: ..."

def construct_prompt(ticker, curday, n_weeks):

    try:
        steps = [n_weeks_before(curday, n) for n in range(n_weeks + 1)][::-1]
    except Exception:
        raise gr.Error(f"Invalid date {curday}!")

    data = stock_data
    data = get_news(ticker, data)
    # data['Basics'] = [json.dumps({})] * len(data)
    # print(data)

    info, prompt = get_all_prompts_online(ticker, data, curday)

    prompt = B_INST + B_SYS + SYSTEM_PROMPT + E_SYS  + E_INST
    # print(prompt)

    return info,prompt
# info, prompt = construct_prompt(ticker, date, n_weeks, use_basics)




construct_prompt(ticker="utkarshbnk", curday=datetime.today().strftime('%Y-%m-%d'), n_weeks=1)

def predict(ticker, date, n_weeks):

    print_gpu_utilization()

    info,prompt = construct_prompt(ticker, date, n_weeks)

    inputs = tokenizer(
        prompt, return_tensors='pt', padding=False
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    print("Inputs loaded onto devices.")

    res = model.generate(
        **inputs, max_length=4096, do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True, streamer=streamer
    )
    output = tokenizer.decode(res[0], skip_special_tokens=True)
    answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL)

    torch.cuda.empty_cache()

    return info,answer
predict(ticker="Reliance", date='2024-02-20', n_weeks=2)

demo = gr.Interface(
    predict,
    inputs=[
        gr.Textbox(
            label="Ticker",
            value="AAPL",
            info="Companys from Dow-30 are recommended"
        ),
        gr.Textbox(
            label="Date",
            value=get_curday,
            info="Date from which the prediction is made, use format yyyy-mm-dd"
        ),
        gr.Slider(
            minimum=1,
            maximum=4,
            value=3,
            step=1,
            label="n_weeks",
            info="Information of the past n weeks will be utilized, choose between 1 and 4"
        )
    ],
    outputs=[
        gr.Textbox(
            label="Information"
        ),
        gr.Textbox(
            label="Response"
        )
    ],
    title="FinGPT-Forecaster",
    description="""FinGPT-Forecaster takes random market news and optional basic financials related to the specified company from the past few weeks as input and responds with the company's **positive developments** and **potential concerns**. Then it gives out a **prediction** of stock price movement for the coming week and its **analysis** summary.""",
)



demo.launch()
