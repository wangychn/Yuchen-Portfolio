import CartPole from "../../components/Cartpole/Cartpole.jsx";

export default function Projects() {
    console.log("CartPole import is:", CartPole);

    return (
        <div>
            <h1>Cart-Pole Demo</h1>
            <CartPole />
        </div>
    );
}