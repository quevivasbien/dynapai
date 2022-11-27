use downcast_rs::{Downcast, impl_downcast};
use dyn_clone::{DynClone, clone_trait_object};
use numpy::ndarray::{Array, Ix1};

use crate::prelude::*;

pub trait PayoffFunc<A: ActionType>: DynClone + Downcast + MutatesOn<A> + Send + Sync {
    fn n(&self) -> usize;
    fn u_i(&self, i: usize, actions: &A) -> f64;
    fn u(&self, actions: &A) -> Array<f64, Ix1> {
        Array::from_iter((0..actions.n()).map(|i| self.u_i(i, actions)))
    }
}

clone_trait_object!(<A> PayoffFunc<A> where A: ActionType);
impl_downcast!(PayoffFunc<A> where A: ActionType);

#[derive(Clone)]
pub struct ModularPayoff<A: ActionType + Clone>
{
    pub n: usize,
    pub prod_func: Box<dyn ProdFunc<A>>,
    pub risk_func: Box<dyn RiskFunc>,
    pub csf: Box<dyn CSF>,
    pub reward_func: Box<dyn RewardFunc>,
    pub disaster_cost: Box<dyn DisasterCost>,
    pub cost_func: Box<dyn CostFunc<A>>,
}

impl<A: ActionType + Clone + 'static> ModularPayoff<A>
{
    pub fn new(
        prod_func: Box<dyn ProdFunc<A>>,
        risk_func: Box<dyn RiskFunc>,
        csf: Box<dyn CSF>,
        reward_func: Box<dyn RewardFunc>,
        disaster_cost: Box<dyn DisasterCost>,
        cost_func: Box<dyn CostFunc<A>>
    ) -> Result<ModularPayoff<A>, &'static str> {
        let n = prod_func.n();
        if n != risk_func.n()
            || n != reward_func.n()
            || n != disaster_cost.n()
            || n != cost_func.n()
        {
            return Err("When creating new ModularPayoff: All components must have the same n");
        }
        Ok(ModularPayoff {
            n,
            prod_func,
            risk_func,
            csf,
            reward_func,
            disaster_cost,
            cost_func,
        })
    }
}

impl<A: ActionType + Clone> MutatesOn<A> for ModularPayoff<A> {
    fn mutate_on(&mut self, actions: &A) {
        self.prod_func.as_mut().mutate_on(actions);
    }
}

impl<A: ActionType + Clone + 'static> PayoffFunc<A> for ModularPayoff<A>
{
    fn n(&self) -> usize {
        self.n
    }

    fn u_i(&self, i: usize, actions: &A) -> f64 {
        let (s, p) = self.prod_func.f(actions);

        let sigmas = self.risk_func.sigma(s.view(), p.view());
        let qs = self.csf.q(p.view());
        let rewards = self.reward_func.reward(i, p.view());
        // payoff given no disaster * proba no disaster
        let no_d = sigmas.iter().zip(qs.iter()).zip(rewards.iter()).map(
            |((sigma, q), reward)| sigma * q * reward
        ).sum::<f64>();
        // cost given disaster * proba disaster
        let yes_d = (1.0 - sigmas.iter().zip(qs.iter()).map(
            |(sigma, q)| sigma * q
        ).sum::<f64>()) * self.disaster_cost.d_i(i, s.view(), p.view());

        no_d - yes_d - self.cost_func.c_i(i, actions)
    }

    fn u(&self, actions: &A) -> Array<f64, Ix1> {
        let (s, p) = self.prod_func.f(actions);
        let sigmas = self.risk_func.sigma(s.view(), p.view());
        let qs = self.csf.q(p.view());

        let all_rewards = (0..p.len()).map(
            |i| self.reward_func.reward(i, p.view())
        );

        let no_d = all_rewards.map(
            |rewards| sigmas.iter().zip(qs.iter()).zip(rewards.iter()).map(
                |((sigma, q), reward)| sigma * q * reward
            ).sum::<f64>()
        );

        let proba_d = 1.0 - sigmas.iter().zip(qs.iter()).map(
            |(sigma, q)| sigma * q
        ).sum::<f64>();

        let disaster_costs = self.disaster_cost.d(s.view(), p.view());
        let yes_d = disaster_costs.iter().map(|d| d * proba_d);

        let net_rewards = no_d.zip(yes_d).map(|(n, y)| n - y);

        let cost = self.cost_func.c(actions);

        Array::from_iter(net_rewards.zip(cost.iter()).map(|(r, c)| r - c))
    }
}
