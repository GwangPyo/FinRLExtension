import matplotlib.pyplot as plt
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import pandas as pd
from copy import deepcopy
from typing import Literal, Sequence
import numpy as np


class StockTradingEnvExtension(StockTradingEnv):
    def __init__(
            self,
            df: pd.DataFrame,
            hmax: int = 500,
            initial_amount: int = 1000_000,
            num_stock_shares: Sequence[int] | None = None,
            buy_cost_pct: Sequence[float] | float = 0.001,
            sell_cost_pct: Sequence[float] | float = 0.001,
            reward_scaling: float = 1e-4,
            tech_indicator_list: list[str] | Literal['auto'] = 'auto',
            turbulence_threshold=None,
            risk_indicator_col="turbulence",
            make_plots: bool = False,
            print_verbosity=10,
            day=0,
            initial=True,
            previous_state=(),
            model_name="",
            mode="",
            iteration="",
            stop_loss_tolerance: float = 0.8 # 0.8 means short selling when the loss is over 20% => 80% price reach
    ):

        self.stop_loss_tolerance = stop_loss_tolerance

        tomorrow_open = df.loc[1:]['open'].values
        # pad
        last_day = (df.index.max())
        if len(df.tic.unique()) > 1:
            pad = df.loc[last_day]['close'].values
        else:
            pad = np.asarray([df.loc[last_day]['close']])

        df["start"] = np.concatenate([tomorrow_open, pad], axis=-1)


        df_columns = deepcopy(list(df.columns))

        df_columns.remove('day')
        df_columns.remove('date')
        df_columns.remove('tic')

        if previous_state == ():
            previous_state = []

        if tech_indicator_list == 'auto':
            # dataframe other than close, high, low, volume
            # they are not available in real-world scenario when the trading is happening
            tech_indicator_list = df_columns
        # calculate sotck dimension, and set state and action space automatically
        stock_dim = len(df['tic'].unique())
        state_space = 1 + 2 * stock_dim + len(tech_indicator_list) * stock_dim
        action_space = stock_dim
        # casting cost
        if isinstance(buy_cost_pct, float):
            buy_cost_pct = [buy_cost_pct] * stock_dim
        else:
            buy_cost_pct = list(buy_cost_pct)
        if isinstance(sell_cost_pct, float):
            sell_cost_pct = [sell_cost_pct] * stock_dim
        else:
            sell_cost_pct = list(sell_cost_pct)
        if num_stock_shares is None:
            num_stock_shares = [0] * stock_dim

        super().__init__(
            df=df,
            stock_dim=stock_dim,
            hmax=hmax,
            initial_amount=initial_amount,
            num_stock_shares=num_stock_shares,
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            reward_scaling=reward_scaling,
            state_space=state_space,
            action_space=action_space,
            tech_indicator_list=tech_indicator_list,
            turbulence_threshold=turbulence_threshold,
            risk_indicator_col=risk_indicator_col,
            make_plots=make_plots,
            print_verbosity=print_verbosity,
            day=day,
            initial=initial,
            previous_state=previous_state,
            model_name=model_name,
            mode=mode,
            iteration=iteration
        )
        self.avg_buy_price = np.zeros(self.stock_dim)
        self.num_stop_loss = 0

    def reset(
            self,
            *,
            seed=None,
            options=None,
    ):
        self.num_stop_loss = 0
        self.avg_buy_price = np.zeros(self.stock_dim)
        return super().reset(seed=seed, options=options)

    @property
    def share(self) -> np.ndarray:  # amount of asset except cash
        return np.copy(np.asarray(self.state[(1 + self.stock_dim):(1 + 2 * self.stock_dim)]))

    @property
    def price(self) -> np.ndarray:  # current price of the assets
        return np.array(self.state[1: (self.stock_dim + 1)])

    @property
    def vec_asset(self) -> np.ndarray:  # cash + current value of each asset
        return np.concatenate([self.price * self.share, [self.state[0]]], axis=0).copy()

    @property
    def asset(self) -> float:  # total value of the current asset
        return self.vec_asset.sum().item()

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                    self.state[0]
                    + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(
                    self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]
                )
            )
                    - self.asset_memory[0]
            )  # initial_amount is only cash part of our initial asset
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                        (252 ** 0.5)
                        * df_total_value["daily_return"].mean()
                        / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                print(f"total_num_top_loss: {self.num_stop_loss}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                plt.close()

            # Add outputs to logger interface
            # logger.record("environment/portfolio_value", end_total_asset)
            # logger.record("environment/total_reward", tot_reward)
            # logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            # logger.record("environment/total_cost", self.cost)
            # logger.record("environment/total_trades", self.trades)

            return np.asarray(self.state).copy(), self.reward, self.terminal, False, { }

        else:
            actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            actions = actions.astype(
                int
            )  # convert into integer because we can't by fraction of shares
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]
            share_before_selling = self.share

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)

                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            share_after_selling = self.share
            # reset all avg buy price if there is no more share
            self.avg_buy_price = np.where(share_after_selling == 0, 0, self.avg_buy_price)

            for index in buy_index:
                # previous
                prev_total_i = share_after_selling[index] * self.avg_buy_price[index]
                # actual buy amount
                actions[index] = self._buy_stock(index, actions[index])
                # buy amount * price
                delta = actions[index] * self.price[index]
                #  (action[index] == 0 & share_after_selling ==0) can be True. In that case, we should not change avg buy price
                #  otherwise, do moving average
                self.avg_buy_price[index] = np.where(actions[index] + share_after_selling[index] > 0,
                                                     (delta + prev_total_i) / (
                                                                 actions[index] + share_after_selling[index]),
                                                     0
                                                     )
            # do stop_loss
            now = self.share
            stop_loss_index = np.where((now > 0 ) & (self.avg_buy_price * self.stop_loss_tolerance > self.price))[0]
            for index in stop_loss_index:
                self._sell_stock(index, now[index])
                self.avg_buy_price[index] = 0
                self.num_stop_loss += 1

            self.actions_memory.append(actions)

            # state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling
            self.state_memory.append(
                self.state
            )  # add current state in state_recorder for each step

        return np.asarray(self.state).copy(), self.reward, self.terminal, False, { }

    def dict_obs(self):
        data = self.df.query(f'index == {self.day}')

        tech = [data[tech].tolist()
                for tech in self.tech_indicator_list]

        # Get unique tickers
        tickers = self.df['tic'].unique()
        obs = { "Cash": self.state[0] , "date": self.df.query(f'index == {self.day}')['date'].values[0]}

        for i, tic in enumerate(tickers):
            obs[tic] = {
                "price (previous close)": self.price[i].item(),
                "share": self.share[i].item(),
                "avg_price": self.avg_buy_price[i].item(),
                "tech": [(self.tech_indicator_list[j], tech_values[i]) for j, tech_values in enumerate(tech)]
            }

        return obs



if __name__ == '__main__':
    from pprint import pprint
    df = pd.read_csv("C:\\Users\\necro\\finRLextension\\finRLext\\finrl\\train_df.csv",
                     )
    df = df.set_index(df.columns[0])
    df = df[df.tic=='JNK']
    env = StockTradingEnvExtension(df, day=500, stop_loss_tolerance=0.91)
    obs, _ = env.reset()
    print(obs)

    for _ in range(500): # do nothing for 500 days
        obs, reward, done, timeout, info = env.step(np.array([0])) # , 0, 0, 0, 0, 0, 0, 0, 0]))
    obs, reward, done, timeout, info = env.step(np.array([1.])) # 0, 0, 1.0, 0, 0, 0, 0, 0]) )
    # 금리 인상기
    for _ in range(200):
        obs, reward, done, timeout, info = env.step(np.array([0]))
        print(reward)
        print(done)
        pprint(env.dict_obs())










