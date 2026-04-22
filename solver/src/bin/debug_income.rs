use machi_koro_solver::coast::expected_income;
use machi_koro_solver::state::{AState, lm, NUM_CARDS};

fn main() {
    let mut cards = [0u8; NUM_CARDS];
    cards[5] = 3; // cheese factory x3
    cards[1] = 3; // ranch x3  →  roll 7: income = 27 (capped at 52)

    let s_train = AState { coins: 0, cards, landmarks: lm::TRAIN };
    let s_park  = AState { coins: 0, cards, landmarks: lm::TRAIN | lm::PARK };
    let s_tower = AState { coins: 0, cards, landmarks: lm::TRAIN | lm::TOWER };
    let s_all   = AState { coins: 0, cards, landmarks: lm::TRAIN | lm::PARK | lm::TOWER };

    let e_train = expected_income(&s_train);
    let e_park  = expected_income(&s_park);
    let e_tower = expected_income(&s_tower);
    let e_all   = expected_income(&s_all);

    println!("train only:        {e_train:.4}");
    println!("train+park:        {e_park:.4}  (expect {:.4} = e_train × 6/5)", e_train * 6.0/5.0);
    println!("train+tower:       {e_tower:.4}");
    println!("train+park+tower:  {e_all:.4}");
    println!("park/train ratio:  {:.4}  (expect 1.2000)", e_park / e_train);
}
