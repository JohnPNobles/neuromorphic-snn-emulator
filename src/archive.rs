// mod tenrules;

// use tenrules::tenrules;

// fn main() {
//     // Welcome messages
//     println!("Hello, world!\n");
//     println!("Welcome to my Rust project!\n");

//     // Addition: within print statement vs. using an outside function

//     // Within print statement:
//     println!("3 + 3 = {} <-- addition inside a print statement\n", 3 + 3);

//     // Using an outside function:
//     let result: i32 = add(3, 3);
//     println!(
//         "3 + 3 = {} <-- addition using an outside function\n",
//         result
//     );

//     // Ten rules of Rust (especially for Python/R users):
//     println!("Ten rules of Rust (especially for Python/R users):\n");
//     tenrules();
// }

// fn add(x: i32, y: i32) -> i32 {
//     x + y
//     // This function takes two integers and returns their sum.

//     /* In Rust, i32 is the default integer type, handling values up to 2.1 billion. i64 handles values up to 9.2 quintillion, but it's more memory-intensive, though on 64-bit systems the performance is virtually equal, and i64 isn't often used anyway since i32 is sufficient for most use cases. */
//     /* Creating functions like this is like creating a mapping for a function that we know from mathematics. Here, we say that x belongs to the set of "reasonable" integers, y belongs to the same set (i32), and the function "add" is a mapping from the Cartesian product of the set of "reasonable" integers to the set of "reasonable" integers (under the operation of addition). */
// }
