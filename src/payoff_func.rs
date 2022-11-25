use dyn_clone::DynClone;
use numpy::ndarray::{Array, Ix1};

use crate::prelude::*;

pub trait PayoffFunc: DynClone + Send + Sync {
    type Act: ActionType;
    fn n(&self) -> usize;
    fn u_i(&self, i: usize, actions: &Self::Act) -> f64;
    fn u(&self, actions: &Self::Act) -> Array<f64, Ix1> {
        Array::from_iter((0..actions.n()).map(|i| self.u_i(i, actions)))
    }
}

impl<A: ActionType> Clone for Box<dyn PayoffFunc<Act = A>> {
    fn clone(&self) -> Self {
        dyn_clone::clone_box(&**self)
    }
}

impl<A: ActionType> PayoffFunc for Box<dyn PayoffFunc<Act = A>> {
    type Act = A;
    fn n(&self) -> usize { self.as_ref().n() }
    fn u_i(&self, i: usize, actions: &A) -> f64 { self.as_ref().u_i(i, actions) }
    fn u(&self, actions: &A) -> Array<f64, Ix1> { self.as_ref().u(actions) }
}


#[derive(Clone)]
pub struct DefaultPayoff<A, T, U, V, W, X, Y>
where A: ActionType,
      T: ProdFunc<A>,
      U: RiskFunc,
      V: CSF,
      W: RewardFunc,
      X: DisasterCost,
      Y: CostFunc<A>,
{
    pub n: usize,
    pub prod_func: T,
    pub risk_func: U,
    pub csf: V,
    pub reward_func: W,
    pub disaster_cost: X,
    pub cost_func: Y,
    _phantom: std::marker::PhantomData<A>,
}

impl<A, T, U, V, W, X, Y> DefaultPayoff<A, T, U, V, W, X, Y>
where A: ActionType,
      T: ProdFunc<A>,
      U: RiskFunc,
      V: CSF,
      W: RewardFunc,
      X: DisasterCost,
      Y: CostFunc<A>,
{
    pub fn new(
        prod_func: T, risk_func: U, csf: V, reward_func: W, disaster_cost: X, cost_func: Y
    ) -> Result<DefaultPayoff<A, T, U, V, W, X, Y>, &'static str> {
        let n = prod_func.n();
        if n != risk_func.n()
            || n != reward_func.n()
            || n != disaster_cost.n()
            || n != cost_func.n()
        {
            return Err("When creating new DefaultPayoff: All components must have the same n");
        }
        Ok(DefaultPayoff {
            n,
            prod_func,
            risk_func,
            csf,
            reward_func,
            disaster_cost,
            cost_func,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<A, T, U, V, W, X, Y> PayoffFunc for DefaultPayoff<A, T, U, V, W, X, Y>
where A: ActionType + Clone,
      T: ProdFunc<A> + Clone,
      U: RiskFunc + Clone,
      V: CSF + Clone,
      W: RewardFunc + Clone,
      X: DisasterCost + Clone,
      Y: CostFunc<A> + Clone,
{
    type Act = A;
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


pub type ModularPayoff<A> = DefaultPayoff<
    A,
    Box<dyn ProdFunc<A>>,
    Box<dyn RiskFunc>,
    Box<dyn CSF>,
    Box<dyn RewardFunc>,
    Box<dyn DisasterCost>,
    Box<dyn CostFunc<A>>,
>;

fn hello(x: &Box<ModularPayoff<Actions>>, actions: &Actions) {
    println!("{:?}", x.u(actions));
}