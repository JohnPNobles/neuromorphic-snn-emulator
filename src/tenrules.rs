pub fn tenrules() {
    println!("\t1. Variables are immutable by default. Use 'mut' to make them mutable.\n");
    println!(
        "\t2. Rust does not have a garbage collector. It uses ownership and borrowing to manage memory.\n"
    );
    println!(
        "\t3. Data can only have one owner at a time. Assigning one variable to another makes the initial one empty, transferring the data from one variable to another.\n"
    );
    println!(
        "\t4. References must be strict. You can either have one mutable reference or any number of immutable references, but not both at the same time.\n"
    );
    println!(
        "\t5. The compiler is your friend. It will help you catch errors quicker than Python/R and guide you to write better code.\n"
    );
    println!(
        "\t6. Everything has a strict type. Rust is a statically typed language, which means that all types must be known at compile time.\n",
    );
    println!(
        "\t7. No null/NA values. Rust uses the Option type to handle cases where a value might be absent, which forces you to handle such cases explicitly.\n",
    );
    println!(
        "\t8. Errors are data. Rust uses the Result type to handle errors, which encourages you to handle errors explicitly rather than ignoring them.\n"
    );
    println!(
        "\t9. Expressions vs. statements. In Rust, most things are expressions, which means they return a value. This allows for more flexible and powerful code. Also, leaving the semicolon off the last line of a function will return the value of that line, which is a common pattern in Rust.\n"
    );
    println!(
        "\t10. Arrays are fixed, vectors are flexible. Arrays have a fixed size, while vectors can grow and shrink in size. Use vectors when you need a dynamic array.\n"
    );
}
