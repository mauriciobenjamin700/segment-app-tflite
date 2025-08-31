import * as RN from "react-native";
import styles from "./styles";

interface ButtonProps {
  title: string;
  onPress: () => void;
}

export default function Button({ title, onPress }: ButtonProps) {
  return (
    <RN.Pressable onPress={onPress} style={styles.button}>
      <RN.Text style={styles.buttonText}>{title}</RN.Text>
    </RN.Pressable>
  );
}
