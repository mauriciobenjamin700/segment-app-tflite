import * as RN from "react-native";
import styles from "./styles";

interface ImagePreviewProps{
    uri?: string;
}

export default function ImagePreview({ uri }: ImagePreviewProps) {
  return (
    <RN.View style={styles.container}>
      <RN.Image
        source={{ uri: uri || "https://via.placeholder.com/150" }}
        style={styles.image}
      />
    </RN.View>
  );
}
