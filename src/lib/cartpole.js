
export class CartPoleEnv {
    constructor({
        gravity = 9.8,
        massCart = 1.0,
        massPole = 0.1,
        length = 0.5,         // actually half the pole’s length
        forceMag = 10.0,
        tau = 0.02,           // time step
        thetaThreshold = 360 * Math.PI / 180,
        xThreshold = 2.4,
    } = {}) {
        // physics params
        this.gravity = gravity;
        this.massCart = massCart;
        this.massPole = massPole;
        this.totalMass = massCart + massPole;
        this.poleMassLength = massPole * length;
        this.length = length;
        this.forceMag = forceMag;
        this.tau = tau;
        // termination thresholds
        this.thetaThreshold = thetaThreshold;
        this.xThreshold = xThreshold;
        // state vector [x, xDot, theta, thetaDot]
        this.state = null;
    }


    reset() {
        // uniform random in [-0.05, +0.05]
        this.state = Array(4).fill(0).map(() => (Math.random() * 0.1 - 0.05));
        return this.state;
    }

    step(action) {
        // apply physics update exactly as Gym’s CartPole
        let [x, xDot, theta, thetaDot] = this.state;

        const force = action === 1 ? this.forceMag : -this.forceMag;
        const costheta = Math.cos(theta);
        const sintheta = Math.sin(theta);

        // equations of motion
        const temp = (force + this.poleMassLength * thetaDot * thetaDot * sintheta)
            / this.totalMass;
        const thetaAcc = (this.gravity * sintheta - costheta * temp) /
            (this.length * (4.0 / 3.0 - this.massPole * costheta * costheta / this.totalMass));
        const xAcc = temp - this.poleMassLength * thetaAcc * costheta / this.totalMass;

        // Euler integration
        x += this.tau * xDot;
        xDot += this.tau * xAcc;
        theta += this.tau * thetaDot;
        thetaDot += this.tau * thetaAcc;

        this.state = [x, xDot, theta, thetaDot];

        // check termination
        const done = (
            x < -this.xThreshold ||
            x > this.xThreshold ||
            theta < -this.thetaThreshold ||
            theta > this.thetaThreshold
        );

        const reward = done ? 0 : 1;
        return { state: this.state, reward, done };
    }
}

export class CartPoleRenderer {
    constructor(canvas, env) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.env = env;
        // drawing params
        this.worldWidth = env.xThreshold * 2;
        this.scale = canvas.width / this.worldWidth;
        this.cartWidth = 120;
        this.cartHeight = 70;
        this.poleWidth = 24;
        // this.poleLen = this.scale * (1.7 * env.length);
        this.poleLen = 280;
    }

    draw(state) {
        const [x, , theta] = state;
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // compute cart center
        const cartX = x * this.scale + this.canvas.width / 2;
        const cartY = this.canvas.height * 0.8;

        // draw cart
        ctx.fillStyle = "#333";
        ctx.fillRect(
            cartX - this.cartWidth / 2,
            cartY - this.cartHeight / 2,
            this.cartWidth,
            this.cartHeight
        );

        // draw pole
        ctx.save();
        ctx.translate(cartX, cartY - this.cartHeight / 2);
        ctx.rotate(theta);
        ctx.fillStyle = "#CA9865";
        ctx.fillRect(
            -this.poleWidth / 2,
            -this.poleLen + this.poleWidth / 2,
            this.poleWidth,
            this.poleLen
        );
        ctx.restore();
    }
}