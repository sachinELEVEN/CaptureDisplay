# ---------------------------------------
# Step 1: Clear dist folder and create CaptureDisplay.app
# ---------------------------------------
#Use the below command to create the spec file, once that is ready use spec file with pyinstaller
#pyinstaller --onefile --windowed ../main.py --name "CaptureDisplay" --icon=../assets/capturedisplay.icns
cd build-pipeline
rm -r dist
rm -r build
#create a bundle identifer from appstoreconnect and use that in the spec file
pyinstaller CaptureDisplay.spec
cd dist
find CaptureDisplay.app -name .DS_Store -delete
cd ..

# ---------------------------------------
# Step 2: Signing the CaptureDisplay.app
# ---------------------------------------
#Signing needs to happen with a certificate- if you use developer id certificate like the one here the notarization does not work, we need to use self signed certificate like 'CaptureDisplay'. Example developer application id certificate -> EC24DE91843FE9267B360FA70CAFAF873E92AC72 "Developer ID Application: sachin jeph (34XRA32US9)"
#Use this command to show the list of available certificates -> security find-identity -p basic -v
#-s hash, this hash is corresponding to the developer id application certificate, you see you current certificates and hashes using the command $security find-identity -p basic -v. We need a developer id application certificate to notarize the apps
#signing the .app where -s denotes teh signture certificate we need to provide. Please use self signed certificate otherwise problems occur in notarization wher notarization fails
#You can create a self signed certificate from Keychain access. Make sure to open the certiciate -> Trust section -> Set Code Signing as 'Always Trust'
#Or okay i just realised you can use your developer certificate and open the certiciate -> Trust section -> Set Code Signing as 'Always Trust' as well, and that should work as well for code signing certificates
#I am just thinkning the code signing using developer certificate might be better than self signed certicate no? i am not entirely sure about this for now, but for now using developer certiciate for code signing
# codesign --deep --force --verbose -s "CaptureDisplay" "./dist/CaptureDisplay.app"
#we must sign with Developer ID Application certificate in order to notarize the app
codesign -s "EC24DE91843FE9267B360FA70CAFAF873E92AC72" -v --deep --force --timestamp --entitlements entitlements.plist -o runtime "./dist/CaptureDisplay.app"

# ---------------------------------------
# Step 3: Convert the application bundle to a DMG (macOS disk image)
# ---------------------------------------
echo "Creating DMG installer..."
sleep 5

#Visit https://github.com/create-dmg/create-dmg for more information on create-dmg
# Create the DMG
# Ensure you have 'create-dmg' installed. If not, install using 'brew install create-dmg'
create-dmg  --volname "CaptureDisplay" --volicon "/Users/sachinjeph/Desktop/CaptureDisplay/assets/capturedisplay.ico" --icon-size 100 --app-drop-link 425 120 "./dist/CaptureDisplay.dmg" "./dist/CaptureDisplay.app/"
# ---------------------------------------
# Step 4: Signing the CaptureDisplay.dmg
# ---------------------------------------
#Signing the dmg
echo "Signing DMG..."
codesign -s "EC24DE91843FE9267B360FA70CAFAF873E92AC72" -v --deep --force --timestamp --entitlements entitlements.plist -o runtime "./dist/CaptureDisplay.dmg"

echo "Packaging and signing complete. You can find the DMG installer in the dist/ directory."

# ---------------------------------------
# Step 5: Notarising the DMG (macOS disk image)
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
#if notarization fails you can see notary logs which has the reasons for failure using the command where this id is the submission id which you would see when you run the initial notarization command
#xcrun notarytool log 8e03c9e3-219f-48be-a22d-4ba4846490c3 --keychain-profile "CaptureDisplay"

#To see the list of certificates
#security find-identity -p basic -v
#Maybe write some upload script here to upload dmg for customers