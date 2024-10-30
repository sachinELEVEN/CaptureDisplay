# # ---------------------------------------
# # Step 1: Clear dist folder and create CaptureDisplay.app
# # ---------------------------------------
# #Use the below command to create the spec file, once that is ready use spec file with pyinstaller
# #pyinstaller --onefile --windowed ../main.py --name "CaptureDisplay" --icon=../assets/capturedisplay.icns
# rm -r dist
# rm -r build
# #create a bundle identifer from appstoreconnect and use that in the spec file
# pyinstaller CaptureDisplay.spec
# cd dist
# find CaptureDisplay.app -name .DS_Store -delete
# cd ..

# # ---------------------------------------
# # Step 2: Signing the CaptureDisplay.app
# # ---------------------------------------
# #-s hash, this hash is corresponding to the developer id application certificate, you see you current certificates and hashes using the command $security find-identity -p basic -v. We need a developer id application certificate to notarize the apps
# #signing the .app
# # codesign --deep --force --verbose -s "CaptureDisplay" "./dist/CaptureDisplay.app"
# codesign -s "EC24DE91843FE9267B360FA70CAFAF873E92AC72" -v --deep --force --timestamp --entitlements entitlements.plist -o runtime "./dist/CaptureDisplay.app"

# # ---------------------------------------
# # Step 3: Convert the application bundle to a DMG (macOS disk image)
# # ---------------------------------------
# echo "Creating DMG installer..."
# sleep 5

# #Visit https://github.com/create-dmg/create-dmg for more information on create-dmg
# # Create the DMG
# # Ensure you have 'create-dmg' installed. If not, install using 'brew install create-dmg'
# create-dmg  --volname "CaptureDisplay" --volicon "/Users/sachinjeph/Desktop/CaptureDisplay/assets/capturedisplay.ico" --icon-size 100 --app-drop-link 425 120 "./dist/CaptureDisplay.dmg" "./dist/BrainSphere.app/"
# # ---------------------------------------
# # Step 4: Signing the CaptureDisplay.dmg
# # ---------------------------------------
# #Signing the dmg
# echo "Signing DMG..."
# codesign -s "EC24DE91843FE9267B360FA70CAFAF873E92AC72" -v --deep --force --timestamp --entitlements entitlements.plist -o runtime "./dist/CaptureDisplay.dmg"

# echo "Packaging and signing complete. You can find the DMG installer in the dist/ directory."

# ---------------------------------------
# Step 2: Notarising the DMG (macOS disk image)
# ---------------------------------------
#For performing notarization you will need a keychain profile, to create it you can run the below command and then on screen steps
#xcrun notarytool store-credentials
#Here you will be asked to create an app specific password which you can generate from account.appleid.com, its simple
#Once your keychain profile is ready it will be there in keychain
echo "Notarising DMG..."
#Notarisation step- takes some time so thats why they are commented
xcrun notarytool submit "./dist/CaptureDisplay.dmg" --keychain-profile "CaptureDisplay" --wait
#Staple the dmg, this allows for app verification without internet access on user's side
xcrun stapler staple "./dist/CaptureDisplay.dmg"
#Validate stapler step
#xcrun stapler validate "./dist/CaptureDisplay.dmg"
#To verify if a dmg is notarised or not
# spctl -a -vvv -t install "./dist/CaptureDisplay.dmg"
#To see history of your notarization request run the below command
# xcrun notarytool history --keychain-profile "CaptureDisplay"
#To get the detailed log of notarization step use the below commnad
#xcrun notarytool log "./dist/CaptureDisplay.dmg" --keychain-profile "CaptureDisplay"
#To see the list of certificates
#security find-identity -p basic -v
#Maybe write some upload script here to upload dmg for customers