const RAD_TO_DEG = 180 / Math.PI;

function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function buttonValue(gamepad, index) {
  return finiteNumber(gamepad?.buttons?.[index]?.value);
}

function buttonPair(gamepad, index) {
  const button = gamepad?.buttons?.[index];
  return [finiteNumber(button?.value), Boolean(button?.touched)];
}

function axisValue(gamepad, index) {
  return finiteNumber(gamepad?.axes?.[index]);
}

function eulerDegrees(orientation) {
  const quaternionX = finiteNumber(orientation.x);
  const quaternionY = finiteNumber(orientation.y);
  const quaternionZ = finiteNumber(orientation.z);
  const quaternionW = finiteNumber(orientation.w, 1);
  const matrix11 = 1 - 2 * (quaternionY * quaternionY + quaternionZ * quaternionZ);
  const matrix12 = 2 * (quaternionX * quaternionY - quaternionZ * quaternionW);
  const matrix13 = 2 * (quaternionX * quaternionZ + quaternionY * quaternionW);
  const matrix22 = 1 - 2 * (quaternionX * quaternionX + quaternionZ * quaternionZ);
  const matrix23 = 2 * (quaternionY * quaternionZ - quaternionX * quaternionW);
  const matrix32 = 2 * (quaternionY * quaternionZ + quaternionX * quaternionW);
  const matrix33 = 1 - 2 * (quaternionX * quaternionX + quaternionY * quaternionY);
  const rotationY = Math.asin(Math.max(-1, Math.min(1, matrix13)));
  let rotationX;
  let rotationZ;

  if (Math.abs(matrix13) < 0.9999999) {
    rotationX = Math.atan2(-matrix23, matrix33);
    rotationZ = Math.atan2(-matrix12, matrix11);
  } else {
    rotationX = Math.atan2(matrix32, matrix22);
    rotationZ = 0;
  }

  return {
    x: rotationX * RAD_TO_DEG,
    y: rotationY * RAD_TO_DEG,
    z: rotationZ * RAD_TO_DEG,
  };
}

function posePayload(transform, hand) {
  const position = transform.position;
  const orientation = transform.orientation;
  const payload = {
    position: {
      x: finiteNumber(position.x),
      y: finiteNumber(position.y),
      z: finiteNumber(position.z),
    },
    rotation: eulerDegrees(orientation),
    quaternion: {
      x: finiteNumber(orientation.x),
      y: finiteNumber(orientation.y),
      z: finiteNumber(orientation.z),
      w: finiteNumber(orientation.w, 1),
    },
  };

  if (hand) {
    payload.hand = hand;
  }
  return payload;
}

function handJoy(gamepad) {
  return {
    axes: [
      axisValue(gamepad, 2),
      axisValue(gamepad, 3),
      buttonValue(gamepad, 0),
      buttonValue(gamepad, 1),
    ],
    buttons: [3, 4, 5].map((index) => buttonPair(gamepad, index)),
  };
}

export function buildQuestPacket({ timestamp, head, leftHand, rightHand, leftGamepad, rightGamepad }) {
  const leftJoy = handJoy(leftGamepad);
  const rightJoy = handJoy(rightGamepad);
  return {
    timestamp: finiteNumber(timestamp, Date.now()),
    LeftHand: posePayload(leftHand, 'left'),
    RightHand: posePayload(rightHand, 'right'),
    Head: posePayload(head),
    Joy: {
      axes: [...leftJoy.axes, ...rightJoy.axes],
      buttons: [...leftJoy.buttons, ...rightJoy.buttons],
    },
  };
}
