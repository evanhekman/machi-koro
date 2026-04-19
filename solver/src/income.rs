use crate::state::{AState, MAX_COINS, card, lm};

#[inline]
pub fn calc_income(state: &AState, roll: u8) -> u8 {
    let c = &state.cards;
    let has_mall = state.landmarks & lm::MALL != 0;
    let income: u16 = match roll {
        1 => c[card::WHEAT_FIELD] as u16,
        2 => {
            c[card::RANCH] as u16
                + c[card::BAKERY] as u16 * if has_mall { 2 } else { 1 }
        }
        3 => c[card::BAKERY] as u16 * if has_mall { 2 } else { 1 },
        4 => c[card::CONVENIENCE_STORE] as u16 * if has_mall { 4 } else { 3 },
        5 => c[card::FOREST] as u16,
        7 => c[card::CHEESE_FACTORY] as u16 * 3 * c[card::RANCH] as u16,
        8 => {
            c[card::FURNITURE_FACTORY] as u16
                * 3
                * (c[card::FOREST] + c[card::MINE]) as u16
        }
        9 => c[card::MINE] as u16 * 5,
        10 => c[card::APPLE_ORCHARD] as u16 * 3,
        11 | 12 => {
            c[card::FRUIT_VEG_MARKET] as u16
                * 2
                * (c[card::WHEAT_FIELD] + c[card::APPLE_ORCHARD]) as u16
        }
        _ => 0,
    };
    income.min(MAX_COINS as u16) as u8
}
